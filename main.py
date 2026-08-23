"""
main.py — E-mboni FastAPI Backend
Database: Firebase Cloud Firestore (migrated from SQLAlchemy/PostgreSQL)
All endpoints, response formats, and auth behavior preserved.
"""

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List
import cv2
import numpy as np
import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("emboni")

from ultralytics import YOLO

from database import (
    init_db, get_db, verify_password, hash_password,
    RoleEnum, AlertLevelEnum, SessionStatusEnum, LanguageEnum, VoiceSpeedEnum, StatusEnum,
)
from auth import create_token, decode_token, get_current_user_from_token
from models import (
    RegisterRequest, RegisterResponse, UserOut, BlindUserOut,
    LoginRequest, LoginResponse, LoginUserOut, BlindUserSummary,
    DetectedObject, DetectionResult,
    AlertOut, SessionOut, UserAdminOut, GuardianDashboardOut,
)
from spatial_engine import RawDetection, process_detections, ALL_OBJECTS
from events import event_store
from memory import CrowdDetector, ConsistencyFilter
from datetime import datetime, timezone, date
import firestore_service as fs

_crowd_detector = CrowdDetector()
_consistency    = ConsistencyFilter()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("E-mboni backend started. Firestore ready.")
    yield

app = FastAPI(
    title="E-mboni API",
    version="1.0.0",
    description="""
## E-mboni — AI Mobility Assistant for Visually Impaired Users

This API powers the E-mboni mobile app. It handles:
- **Authentication** — register and login for guardian + blind user pairs
- **Detection** — real-time object detection from camera frames
- **Alerts** — danger alerts saved to database and visible to guardian
- **Sessions** — navigation session tracking
- **Guardian Dashboard** — privacy-safe view of blind user activity
- **Admin** — user management

### How to authenticate
1. Call `POST /auth/login` with phone + password
2. Copy the `token` from the response
3. Click **Authorize** (top right) and enter: `Bearer <token>`
4. All protected endpoints will now work

### Demo accounts
| Role | Phone | Password |
|---|---|---|
| admin | +250780000000 | admin123 |
| guardian | +250781000001 | guardian123 |
| blind | +250781000002 | blind123 |
    """,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

custom_model = YOLO("yolov8n.onnx", task="detect")
general_model = YOLO("yolov8n.pt",   task="detect")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _require_auth(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    token = authorization.split(" ", 1)[1]
    return get_current_user_from_token(token)


def _require_role(role: RoleEnum):
    def checker(current_user: dict = Depends(_require_auth)):
        if current_user["role"] != role.value:
            raise HTTPException(status_code=403, detail="Access denied.")
        return current_user
    return checker


def _user_to_admin_out(u: dict) -> UserAdminOut:
    return UserAdminOut(
        id=u["id"],
        name=u["name"],
        phone=u["phone"],
        role=u["role"],
        language=u.get("language", "en"),
        voice_speed=u.get("voice_speed", "Normal"),
        status=u.get("status", "active"),
        guardian_id=u.get("guardian_id"),
        created_at=u["created_at"],
    )


def _alert_to_out(a: dict) -> AlertOut:
    return AlertOut(
        id=a["id"],
        blind_id=a["blind_id"],
        message=a["message"],
        level=a["level"],
        is_read=a.get("is_read", False),
        created_at=a["created_at"],
    )


def _session_to_out(s: dict) -> SessionOut:
    return SessionOut(
        id=s["id"],
        blind_id=s["blind_id"],
        started_at=s["started_at"],
        ended_at=s.get("ended_at"),
        status=s["status"],
    )


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=RegisterResponse, status_code=201, tags=["Authentication"])
async def register(body: RegisterRequest, db=Depends(get_db)):
    import asyncio
    from functools import partial

    for phone in [body.guardian.phone, body.blind_user.phone]:
        if fs.get_user_by_phone(phone):
            raise HTTPException(status_code=422, detail="Phone number already registered")

    loop = asyncio.get_event_loop()
    guardian_hash = await loop.run_in_executor(None, partial(hash_password, body.guardian.password))
    blind_hash    = await loop.run_in_executor(None, partial(hash_password, body.blind_user.phone[-6:]))

    guardian = fs.create_user(
        name=body.guardian.name,
        phone=body.guardian.phone,
        password_hash=guardian_hash,
        role=RoleEnum.guardian.value,
        relationship=body.guardian.relationship,
    )

    blind_user = fs.create_user(
        name=body.blind_user.name,
        phone=body.blind_user.phone,
        password_hash=blind_hash,
        role=RoleEnum.blind.value,
        language=body.blind_user.language,
        voice_speed=body.blind_user.voice_speed,
        emergency_phone=guardian["phone"],
        guardian_id=guardian["id"],
    )

    logger.info(f"REGISTER | guardian={guardian['phone']} blind={blind_user['phone']}")
    token = create_token(guardian["id"], guardian["role"])
    return RegisterResponse(
        guardian=UserOut(id=guardian["id"], name=guardian["name"], phone=guardian["phone"], role=guardian["role"]),
        blind_user=BlindUserOut(id=blind_user["id"], name=blind_user["name"], phone=blind_user["phone"],
                                role=blind_user["role"], guardian_id=guardian["id"]),
        token=token,
    )


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

@app.post("/auth/login", response_model=LoginResponse, tags=["Authentication"])
def login(body: LoginRequest, db=Depends(get_db)):
    user = fs.get_user_by_phone(body.phone)
    if not user or not verify_password(body.password, user["password_hash"]):
        logger.warning(f"LOGIN FAILED | phone={body.phone}")
        raise HTTPException(status_code=401, detail="Wrong phone or password.")

    logger.info(f"LOGIN OK | phone={user['phone']} role={user['role']}")
    blind_summary = None
    if user["role"] == RoleEnum.guardian.value:
        linked_list = fs.get_blind_users_for_guardian(user["id"])
        if linked_list:
            linked = linked_list[0]
            blind_summary = BlindUserSummary(
                id=linked["id"],
                name=linked["name"],
                status=linked.get("status", "active"),
                language=linked.get("language", "en"),
            )

    return LoginResponse(
        token=create_token(user["id"], user["role"]),
        user=LoginUserOut(
            id=user["id"],
            name=user["name"],
            role=user["role"],
            language=user.get("language", "en"),
            voice_speed=user.get("voice_speed", "Normal"),
            blind_user=blind_summary,
        ),
    )


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------

MOVING_OBJECTS = {
    "car", "truck", "bus", "motorcycle", "bicycle", "person",
    "dog", "cat", "vehicle", "van", "scooter", "animal",
}
HIGH_DANGER_OBJECTS   = {"car", "truck", "bus", "motorcycle", "vehicle", "van"}
MEDIUM_DANGER_OBJECTS = {
    "bicycle", "scooter", "person", "dog", "cat", "animal",
    "chair", "table", "bench", "pole", "fire hydrant", "trash can",
    "staircase", "stairs", "step",
}
CRITICAL_CLOSE_OBJECTS = {"car", "stair", "truck", "bus", "motorcycle", "staircase", "stairs"}
DIRECTION_MAP = {
    "to your far left":  "left",
    "on your left":      "left",
    "straight ahead":    "center",
    "on your right":     "right",
    "to your far right": "right",
}


def _norm_area_to_meters(norm_area: float) -> float:
    if norm_area > 0.4:  return 0.5
    if norm_area > 0.2:  return 1.0
    if norm_area > 0.1:  return 2.0
    if norm_area > 0.05: return 3.5
    if norm_area > 0.02: return 6.0
    if norm_area > 0.01: return 10.0
    return 15.0


def _danger_level(name: str, distance: float) -> str:
    if name in CRITICAL_CLOSE_OBJECTS and distance <= 1.5:
        return "danger"
    if name in HIGH_DANGER_OBJECTS:
        return "danger" if distance <= 5 else "warning"
    if name in MEDIUM_DANGER_OBJECTS:
        if distance <= 1.5: return "danger"
        if distance <= 4:   return "warning"
        return "safe"
    return "warning" if distance <= 1 else "safe"


def _map_direction(direction_str: str) -> str:
    base = direction_str.split(",")[0].strip()
    return DIRECTION_MAP.get(base, "center")


def _build_summary(objects: list) -> str:
    if not objects:
        return "Path clear."
    top = objects[0]
    moving = "moving " if top.isMoving else ""
    return f"{moving}{top.name}, {top.distanceMeters:.1f} meters {'ahead' if top.direction == 'center' else top.direction}"


PATH_CLEAR_SILENCE_SECONDS = 5.0


@app.post("/detect", response_model=DetectionResult, tags=["Detection"])
async def detect(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    db=Depends(get_db),
):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file received.")

    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPEG.")

    h, w = img.shape[:2]

    blind_user: Optional[dict] = None
    if authorization and authorization.startswith("Bearer "):
        try:
            payload = decode_token(authorization.split(" ", 1)[1])
            u = fs.get_user_by_id(int(payload["sub"]))
            if u and u["role"] == RoleEnum.blind.value:
                blind_user = u
        except Exception:
            pass

    results_c = custom_model(img, conf=0.6)
    results_g = general_model(img, conf=0.6)

    raw_detections: list[RawDetection] = []
    seen_labels: set[str] = set()

    for r in (results_c + results_g):
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            if label not in ALL_OBJECTS or label in seen_labels:
                continue
            x1, y1, x2, y2 = box.xyxy[0]
            raw_detections.append(RawDetection(
                label=label,
                box_area=float((x2 - x1) * (y2 - y1)),
                norm_x=float(((x1 + x2) / 2) / w),
                norm_y=float(((y1 + y2) / 2) / h),
                box_w=float(x2 - x1),
                box_h=float(y2 - y1),
            ))
            seen_labels.add(label)

    spatial_results = process_detections(raw_detections)

    _label_priorities = {s.object: s.priority for s in spatial_results}
    _consistency.update([r.label for r in raw_detections], _label_priorities)
    raw_detections   = [r for r in raw_detections if _consistency.is_confirmed(r.label)]
    spatial_results  = [s for s in spatial_results if _consistency.is_confirmed(s.object)]

    is_crowd, crowd_message = _crowd_detector.evaluate(raw_detections, spatial_results)

    raw_by_label = {r.label: r for r in raw_detections}
    detected_objects: list[DetectedObject] = []
    for spatial in spatial_results:
        raw = raw_by_label.get(spatial.object)
        if not raw:
            continue
        norm_area = raw.box_area / (w * h)
        dist_m    = _norm_area_to_meters(norm_area)
        direction = _map_direction(spatial.direction)
        detected_objects.append(DetectedObject(
            name=raw.label,
            isMoving=raw.label in MOVING_OBJECTS,
            distanceMeters=dist_m,
            direction=direction,
            dangerLevel=_danger_level(raw.label, dist_m),
        ))

    danger_order = {"danger": 0, "warning": 1, "safe": 2}
    detected_objects.sort(key=lambda o: danger_order[o.dangerLevel])
    top_danger = detected_objects[0].dangerLevel if detected_objects else "safe"

    if blind_user and detected_objects:
        top = detected_objects[0]
        if top.dangerLevel in ("danger", "warning"):
            event_store.log_safety_event(
                hazard=top.name,
                direction=top.direction,
                distance=str(top.distanceMeters),
            )
            top_spatial = next((s for s in spatial_results if s.object == top.name), None)
            fs.create_alert(
                blind_id=blind_user["id"],
                message=top_spatial.speech if top_spatial else top.name,
                level=top.dangerLevel,
            )

    if is_crowd and crowd_message:
        summary = crowd_message
    elif not detected_objects:
        if blind_user:
            last = fs.get_last_alert_for_blind(blind_user["id"])
            if last:
                now = datetime.now(timezone.utc)
                last_time = last["created_at"]
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                seconds_ago = (now - last_time).total_seconds()
                summary = "Path clear." if seconds_ago >= PATH_CLEAR_SILENCE_SECONDS else ""
            else:
                summary = "Path clear."
        else:
            summary = "Path clear."
    else:
        summary = _build_summary(detected_objects)

    logger.info(f"DETECT | user={blind_user['phone'] if blind_user else 'anonymous'} | topDanger={top_danger} | objects={len(detected_objects)}")
    return DetectionResult(objects=detected_objects, summary=summary, topDanger=top_danger)


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------

@app.get("/auth/me", tags=["Authentication"])
def get_me(current_user: dict = Depends(_require_auth)):
    return {
        "id":              current_user["id"],
        "name":            current_user["name"],
        "phone":           current_user["phone"],
        "role":            current_user["role"],
        "language":        current_user.get("language", "en"),
        "voice_speed":     current_user.get("voice_speed", "Normal"),
        "status":          current_user.get("status", "active"),
        "emergency_phone": current_user.get("emergency_phone"),
        "guardian_id":     current_user.get("guardian_id"),
    }


# ---------------------------------------------------------------------------
# /users/*  — admin only
# ---------------------------------------------------------------------------

@app.get("/users", response_model=List[UserAdminOut], tags=["Admin"])
def list_users(current_user: dict = Depends(_require_role(RoleEnum.admin)), db=Depends(get_db)):
    return [_user_to_admin_out(u) for u in fs.get_all_users()]


@app.get("/users/{user_id}", response_model=UserAdminOut, tags=["Admin"])
def get_user(user_id: int, current_user: dict = Depends(_require_role(RoleEnum.admin)), db=Depends(get_db)):
    user = fs.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return _user_to_admin_out(user)


@app.patch("/users/{user_id}/status", tags=["Admin"])
def update_user_status(
    user_id: int,
    status: str,
    current_user: dict = Depends(_require_role(RoleEnum.admin)),
    db=Depends(get_db),
):
    if status not in [s.value for s in StatusEnum]:
        raise HTTPException(status_code=422, detail="Invalid status value.")
    user = fs.update_user_status(user_id, status)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"id": user_id, "status": status}


# ---------------------------------------------------------------------------
# /alerts/*
# ---------------------------------------------------------------------------

@app.get("/alerts", response_model=List[AlertOut], tags=["Alerts"])
def get_alerts(current_user: dict = Depends(_require_auth), db=Depends(get_db)):
    if current_user["role"] == RoleEnum.guardian.value:
        blind_list = fs.get_blind_users_for_guardian(current_user["id"])
        if not blind_list:
            return []
        return [_alert_to_out(a) for a in fs.get_alerts_for_blind(blind_list[0]["id"], limit=50)]

    if current_user["role"] == RoleEnum.admin.value:
        return [_alert_to_out(a) for a in fs.get_all_alerts(limit=100)]

    raise HTTPException(status_code=403, detail="Access denied.")


@app.get("/alerts/{blind_id}", response_model=List[AlertOut], tags=["Alerts"])
def get_alerts_for_blind(
    blind_id: int,
    current_user: dict = Depends(_require_role(RoleEnum.admin)),
    db=Depends(get_db),
):
    return [_alert_to_out(a) for a in fs.get_alerts_for_blind(blind_id)]


# ---------------------------------------------------------------------------
# /guardian/*
# ---------------------------------------------------------------------------

@app.get("/guardian/dashboard", response_model=GuardianDashboardOut, tags=["Guardian"])
def guardian_dashboard(current_user: dict = Depends(_require_role(RoleEnum.guardian)), db=Depends(get_db)):
    blind_list     = fs.get_blind_users_for_guardian(current_user["id"])
    blind          = blind_list[0] if blind_list else None
    recent_alerts  = []
    active_session = None

    if blind:
        blind_ids     = [u["id"] for u in blind_list]
        recent_alerts = [_alert_to_out(a) for a in fs.get_alerts_for_blind_ids(blind_ids, limit=10)]
        sess          = fs.get_active_session(blind["id"])
        active_session = _session_to_out(sess) if sess else None

    return GuardianDashboardOut(
        guardian=_user_to_admin_out(current_user),
        blind_user=_user_to_admin_out(blind) if blind else None,
        recent_alerts=recent_alerts,
        active_session=active_session,
    )


@app.get("/guardian/tracking", tags=["Guardian"])
def guardian_tracking(current_user: dict = Depends(_require_role(RoleEnum.guardian)), db=Depends(get_db)):
    blind_list = fs.get_blind_users_for_guardian(current_user["id"])
    if not blind_list:
        return {"timeline": [], "session": {"status": "No blind user", "duration_minutes": 0, "alert_count": 0}, "blind_name": ""}

    blind  = blind_list[0]
    alerts = fs.get_alerts_for_blind(blind["id"], limit=20)
    sess   = fs.get_active_session(blind["id"])

    duration = 0
    if sess:
        now     = datetime.now(timezone.utc)
        started = sess["started_at"]
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        duration = int((now - started).total_seconds() / 60)

    top_level = "Safe"
    if alerts:
        if alerts[0]["level"] == "danger":
            top_level = "Danger"
        elif alerts[0]["level"] == "warning":
            top_level = "Warning"

    timeline = [
        {"time": a["created_at"].strftime("%H:%M"), "event": a["message"], "level": a["level"]}
        for a in alerts
    ]

    return {
        "blind_name":  blind["name"],
        "blind_phone": blind["phone"],
        "is_scanning": sess is not None,
        "timeline":    timeline,
        "session": {
            "status":           top_level,
            "duration_minutes": duration,
            "alert_count":      len(alerts),
        },
    }


@app.get("/guardian/alerts", response_model=List[AlertOut], tags=["Guardian"])
def guardian_alerts(current_user: dict = Depends(_require_role(RoleEnum.guardian)), db=Depends(get_db)):
    blind_list = fs.get_blind_users_for_guardian(current_user["id"])
    if not blind_list:
        return []
    return [_alert_to_out(a) for a in fs.get_alerts_for_blind(blind_list[0]["id"], limit=50)]


@app.get("/guardian/emergency", tags=["Guardian"])
def guardian_emergency_get(current_user: dict = Depends(_require_role(RoleEnum.guardian)), db=Depends(get_db)):
    blind_list = fs.get_blind_users_for_guardian(current_user["id"])
    if not blind_list:
        raise HTTPException(status_code=404, detail="No blind user linked.")
    blind = blind_list[0]
    return {
        "blind_name":      blind["name"],
        "emergency_phone": blind.get("emergency_phone"),
        "guardian_phone":  current_user["phone"],
    }


@app.post("/guardian/emergency", tags=["Guardian"])
def guardian_emergency_post(current_user: dict = Depends(_require_role(RoleEnum.guardian)), db=Depends(get_db)):
    """Trigger emergency — returns blind user contact info for the Flutter app to initiate a call."""
    blind_list = fs.get_blind_users_for_guardian(current_user["id"])
    if not blind_list:
        raise HTTPException(status_code=404, detail="No blind user linked.")
    blind = blind_list[0]
    logger.info(f"EMERGENCY | guardian={current_user['phone']} blind={blind['phone']}")
    return {
        "blind_name":      blind["name"],
        "emergency_phone": blind.get("emergency_phone"),
        "guardian_phone":  current_user["phone"],
        "triggered":       True,
    }


@app.post("/guardian/alerts/{alert_id}/read", tags=["Guardian"])
def mark_alert_read(
    alert_id: int,
    current_user: dict = Depends(_require_role(RoleEnum.guardian)),
    db=Depends(get_db),
):
    """Mark an alert as read. Guardian can only mark alerts belonging to their blind user."""
    alert = fs.get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    # Verify the alert belongs to this guardian's blind user
    blind_list = fs.get_blind_users_for_guardian(current_user["id"])
    blind_ids  = [u["id"] for u in blind_list]
    if alert["blind_id"] not in blind_ids:
        raise HTTPException(status_code=403, detail="Access denied.")
    updated = fs.mark_alert_read(alert_id)
    return _alert_to_out(updated)


# ---------------------------------------------------------------------------
# /session/*
# ---------------------------------------------------------------------------

@app.post("/session/start", response_model=SessionOut, status_code=201, tags=["Sessions"])
def start_session(current_user: dict = Depends(_require_role(RoleEnum.blind)), db=Depends(get_db)):
    sess = fs.start_session(current_user["id"])
    return _session_to_out(sess)


@app.post("/session/end", response_model=SessionOut, tags=["Sessions"])
def end_session(current_user: dict = Depends(_require_role(RoleEnum.blind)), db=Depends(get_db)):
    sess = fs.end_session(current_user["id"])
    if not sess:
        raise HTTPException(status_code=404, detail="No active session found.")
    return _session_to_out(sess)


@app.get("/session/active", response_model=Optional[SessionOut], tags=["Sessions"])
def get_active_session(current_user: dict = Depends(_require_auth), db=Depends(get_db)):
    blind_id = None
    if current_user["role"] == RoleEnum.blind.value:
        blind_id = current_user["id"]
    elif current_user["role"] == RoleEnum.guardian.value:
        blind_list = fs.get_blind_users_for_guardian(current_user["id"])
        blind_id   = blind_list[0]["id"] if blind_list else None
    if not blind_id:
        return None
    sess = fs.get_active_session(blind_id)
    return _session_to_out(sess) if sess else None


@app.get("/session/history", response_model=List[SessionOut], tags=["Sessions"])
def session_history(current_user: dict = Depends(_require_role(RoleEnum.admin)), db=Depends(get_db)):
    return [_session_to_out(s) for s in fs.get_all_sessions(limit=100)]


# ---------------------------------------------------------------------------
# /admin/*
# ---------------------------------------------------------------------------

@app.get("/admin/overview", tags=["Admin"])
def admin_overview(current_user: dict = Depends(_require_role(RoleEnum.admin)), db=Depends(get_db)):
    all_users       = fs.get_all_users()
    total_users     = len(all_users)
    total_guardians = sum(1 for u in all_users if u["role"] == RoleEnum.guardian.value)

    active_blind_ids = fs.get_active_session_ids()
    active_now       = len(active_blind_ids)

    today_start  = datetime.combine(date.today(), datetime.min.time()).replace(tzinfo=timezone.utc)
    alerts_today = fs.count_alerts_since(today_start)

    active_users = []
    for blind_id in active_blind_ids:
        u = fs.get_user_by_id(blind_id)
        if u:
            active_users.append({"id": u["id"], "name": u["name"], "status": "Scanning"})

    recent_alerts_raw = fs.get_alerts_with_user(limit=10)
    recent_alerts = [
        {
            "id":         a["id"],
            "user_name":  a.get("user_name", "Unknown"),
            "message":    a["message"],
            "level":      a["level"],
            "created_at": a["created_at"].isoformat(),
        }
        for a in recent_alerts_raw
    ]

    return {
        "total_users":     total_users,
        "total_guardians": total_guardians,
        "active_now":      active_now,
        "alerts_today":    alerts_today,
        "active_users":    active_users,
        "recent_alerts":   recent_alerts,
    }


@app.get("/admin/logs", tags=["Admin"])
def admin_logs(current_user: dict = Depends(_require_role(RoleEnum.admin)), db=Depends(get_db)):
    rows = fs.get_alerts_with_user(limit=200)
    return [
        {
            "id":         a["id"],
            "blind_id":   a["blind_id"],
            "user_name":  a.get("user_name", "Unknown"),
            "message":    a["message"],
            "level":      a["level"],
            "created_at": a["created_at"].isoformat(),
        }
        for a in rows
    ]


# ---------------------------------------------------------------------------
# Legacy in-memory endpoints (kept for backward compat)
# ---------------------------------------------------------------------------

@app.get("/location/last-known")
async def last_known_location(user_id: str, authorization: Optional[str] = Header(None)):
    location = event_store.get_last_known_location(user_id)
    return {"location": location, "sharing_enabled": location is not None}


@app.post("/location/share")
async def toggle_location_sharing(user_id: str, enabled: bool):
    event_store.enable_location_sharing(user_id, enabled)
    return {"sharing_enabled": enabled}


@app.get("/devices/status")
async def device_status(device_id: str):
    return {"device_id": device_id, "status": event_store.get_device_status(device_id)}


@app.post("/devices/ping")
async def device_ping(device_id: str):
    event_store.ping(device_id)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
