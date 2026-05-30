"""
main.py — E-mboni FastAPI Backend
"""

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from typing import Optional, List
import cv2
import numpy as np
import uvicorn
import logging

# ---------------------------------------------------------------------------
# Logging — shows every request in terminal during frontend testing
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("emboni")

from ultralytics import YOLO

from database import (
    init_db, get_db, verify_password, hash_password,
    User, Alert, Session as NavSession,
    RoleEnum, AlertLevelEnum, SessionStatusEnum, LanguageEnum, VoiceSpeedEnum, StatusEnum,
)
from auth import create_token, decode_token
from models import (
    RegisterRequest, RegisterResponse, UserOut, BlindUserOut,
    LoginRequest, LoginResponse, LoginUserOut, BlindUserSummary,
    DetectedObject, DetectionResult,
    AlertOut, SessionOut, UserAdminOut, GuardianDashboardOut,
)
from spatial_engine import RawDetection, process_detections, ALL_OBJECTS
from events import event_store
from memory import CrowdDetector, ConsistencyFilter
from datetime import datetime, timezone

# One crowd detector and one consistency filter shared across all requests
_crowd_detector = CrowdDetector()
_consistency = ConsistencyFilter()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("E-mboni backend started. Database ready.")
    yield

# ---------------------------------------------------------------------------
# Swagger metadata — this is what the frontend team sees at /docs
# ---------------------------------------------------------------------------
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

def _require_auth(authorization: str = Header(...), db: Session = Depends(get_db)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


def _require_role(role: RoleEnum):
    def checker(current_user: User = Depends(_require_auth)):
        if current_user.role != role:
            raise HTTPException(status_code=403, detail="Access denied.")
        return current_user
    return checker


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=RegisterResponse, status_code=201, tags=["Authentication"])
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    for phone in [body.guardian.phone, body.blind_user.phone]:
        if db.query(User).filter(User.phone == phone).first():
            raise HTTPException(status_code=422, detail="Phone number already registered")

    guardian = User(
        name=body.guardian.name,
        phone=body.guardian.phone,
        password_hash=hash_password(body.guardian.password),
        role=RoleEnum.guardian,
        relationship=body.guardian.relationship,
    )
    db.add(guardian)
    db.flush()

    blind_user = User(
        name=body.blind_user.name,
        phone=body.blind_user.phone,
        password_hash=hash_password(body.blind_user.phone[-6:]),
        role=RoleEnum.blind,
        language=body.blind_user.language,
        voice_speed=body.blind_user.voice_speed,
        emergency_phone=guardian.phone,
        guardian_id=guardian.id,
    )
    db.add(blind_user)
    db.commit()
    db.refresh(guardian)
    db.refresh(blind_user)

    logger.info(f"REGISTER | guardian={guardian.phone} blind={blind_user.phone}")
    token = create_token(guardian.id, guardian.role.value)
    return RegisterResponse(
        guardian=UserOut(id=guardian.id, name=guardian.name, phone=guardian.phone, role=guardian.role.value),
        blind_user=BlindUserOut(id=blind_user.id, name=blind_user.name, phone=blind_user.phone,
                                role=blind_user.role.value, guardian_id=guardian.id),
        token=token,
    )


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

@app.post("/auth/login", response_model=LoginResponse, tags=["Authentication"])
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == body.phone).first()
    if not user or not verify_password(body.password, user.password_hash):
        logger.warning(f"LOGIN FAILED | phone={body.phone}")
        raise HTTPException(status_code=401, detail="Wrong phone or password.")

    logger.info(f"LOGIN OK | phone={user.phone} role={user.role.value}")
    blind_summary = None
    if user.role == RoleEnum.guardian:
        linked = db.query(User).filter(User.guardian_id == user.id).first()
        if linked:
            blind_summary = BlindUserSummary(
                id=linked.id,
                name=linked.name,
                status=linked.status.value,
                language=linked.language.value,
            )

    return LoginResponse(
        token=create_token(user.id, user.role.value),
        user=LoginUserOut(
            id=user.id,
            name=user.name,
            role=user.role.value,
            language=user.language.value,
            voice_speed=user.voice_speed.value,
            blind_user=blind_summary,
        ),
    )


# ---------------------------------------------------------------------------
# POST /detect  — updated to frontend DetectionResult format
# ---------------------------------------------------------------------------

MOVING_OBJECTS = {
    "car", "truck", "bus", "motorcycle", "bicycle", "person",
    "dog", "cat", "vehicle", "van", "scooter", "animal",
}

HIGH_DANGER_OBJECTS  = {"car", "truck", "bus", "motorcycle", "vehicle", "van"}
MEDIUM_DANGER_OBJECTS = {
    "bicycle", "scooter", "person", "dog", "cat", "animal",
    "chair", "table", "bench", "pole", "fire hydrant", "trash can",
    "staircase", "stairs", "step",
}

# Objects that become DANGER at <= 1.5m regardless of category (safety brain rule)
CRITICAL_CLOSE_OBJECTS = {"car", "stair", "truck", "bus", "motorcycle", "staircase", "stairs"}

DIRECTION_MAP = {
    "to your far left": "left",
    "on your left":     "left",
    "straight ahead":   "center",
    "on your right":    "right",
    "to your far right":"right",
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
    # Safety brain rule: car or stair within 1.5m → always danger, save immediately
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
    # direction_str may include vertical suffix like "on your left, low"
    base = direction_str.split(",")[0].strip()
    return DIRECTION_MAP.get(base, "center")


def _build_summary(objects: list[DetectedObject]) -> str:
    if not objects:
        return "Path clear."
    top = objects[0]
    moving = "moving " if top.isMoving else ""
    return f"{moving}{top.name}, {top.distanceMeters:.1f} meters {'ahead' if top.direction == 'center' else top.direction}"


def _save_alert(db: Session, blind_id: int, message: str, level: str):
    """INSERT a danger or warning alert into the alerts table."""
    db.add(Alert(
        blind_id=blind_id,
        message=message,
        level=AlertLevelEnum(level),
    ))
    db.commit()


def _last_alert_seconds_ago(db: Session, blind_id: int) -> float:
    """
    Returns how many seconds ago the last alert was saved for this blind user.
    Returns 999 if no alerts exist yet (safe to announce path clear).
    """
    last = (
        db.query(Alert)
        .filter(Alert.blind_id == blind_id)
        .order_by(Alert.created_at.desc())
        .first()
    )
    if not last:
        return 999.0
    now = datetime.now(timezone.utc)
    last_time = last.created_at
    # Make timezone-aware if stored as naive UTC
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)
    return (now - last_time).total_seconds()


PATH_CLEAR_SILENCE_SECONDS = 5.0  # must be this long since last alert to say "Path clear"


@app.post("/detect", response_model=DetectionResult, tags=["Detection"])
async def detect(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file received.")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPEG.")

    h, w = img.shape[:2]

    # --- Identify the blind user from token (optional) ---
    blind_user: Optional[User] = None
    if authorization and authorization.startswith("Bearer "):
        try:
            payload = decode_token(authorization.split(" ", 1)[1])
            u = db.query(User).filter(User.id == int(payload["sub"])).first()
            if u and u.role == RoleEnum.blind:
                blind_user = u
        except Exception:
            pass

    # --- Run both YOLO models ---
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

    # --- CONSISTENCY FILTER: only report objects seen 2+ frames in a row ---
    _consistency.update([r.label for r in raw_detections])
    raw_detections = [r for r in raw_detections if _consistency.is_confirmed(r.label)]
    spatial_results = [s for s in spatial_results if _consistency.is_confirmed(s.object)]

    # --- Crowd detection check (runs before individual object logic) ---
    is_crowd, crowd_message = _crowd_detector.evaluate(raw_detections, spatial_results)

    # --- Build frontend-format objects ---
    # Use spatial_results (already filtered + sorted) and look up matching raw by label
    raw_by_label = {r.label: r for r in raw_detections}
    detected_objects: list[DetectedObject] = []
    for spatial in spatial_results:
        raw = raw_by_label.get(spatial.object)
        if not raw:
            continue
        norm_area = raw.box_area / (w * h)
        dist_m = _norm_area_to_meters(norm_area)
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

    # --- Save danger/warning alerts to DB ---
    if blind_user and detected_objects:
        top = detected_objects[0]
        if top.dangerLevel in ("danger", "warning"):
            event_store.log_safety_event(
                hazard=top.name,
                direction=top.direction,
                distance=str(top.distanceMeters),
            )
            top_spatial = next((s for s in spatial_results if s.object == top.name), None)
            _save_alert(
                db=db,
                blind_id=blind_user.id,
                message=top_spatial.speech if top_spatial else top.name,
                level=top.dangerLevel,
            )

    # --- Build summary ---
    # Crowd mode: collapse individual person alerts into one crowd message
    if is_crowd and crowd_message:
        summary = crowd_message
    elif not detected_objects:
        # Path clear: only announce if DB confirms silence for 5+ seconds
        if blind_user:
            seconds_ago = _last_alert_seconds_ago(db, blind_user.id)
            summary = "Path clear." if seconds_ago >= PATH_CLEAR_SILENCE_SECONDS else ""
        else:
            summary = "Path clear."
    else:
        summary = _build_summary(detected_objects)

    logger.info(f"DETECT | user={blind_user.phone if blind_user else 'anonymous'} | topDanger={top_danger} | objects={len(detected_objects)}")
    return DetectionResult(objects=detected_objects, summary=summary, topDanger=top_danger)


# ---------------------------------------------------------------------------
# GET /auth/me  — current logged-in user profile
# ---------------------------------------------------------------------------

@app.get("/auth/me", tags=["Authentication"])
def get_me(current_user: User = Depends(_require_auth)):
    return {
        "id":              current_user.id,
        "name":            current_user.name,
        "phone":           current_user.phone,
        "role":            current_user.role.value,
        "language":        current_user.language.value,
        "voice_speed":     current_user.voice_speed.value,
        "status":          current_user.status.value,
        "emergency_phone": current_user.emergency_phone,
        "guardian_id":     current_user.guardian_id,
    }


# ---------------------------------------------------------------------------
# /users/*  — admin only
# ---------------------------------------------------------------------------

@app.get("/users", response_model=List[UserAdminOut], tags=["Admin"])
def list_users(
    current_user: User = Depends(_require_role(RoleEnum.admin)),
    db: Session = Depends(get_db),
):
    return db.query(User).all()


@app.get("/users/{user_id}", response_model=UserAdminOut, tags=["Admin"])
def get_user(
    user_id: int,
    current_user: User = Depends(_require_role(RoleEnum.admin)),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


@app.patch("/users/{user_id}/status", tags=["Admin"])
def update_user_status(
    user_id: int,
    status: str,
    current_user: User = Depends(_require_role(RoleEnum.admin)),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    try:
        user.status = StatusEnum(status)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid status value.")
    db.commit()
    return {"id": user_id, "status": user.status.value}


# ---------------------------------------------------------------------------
# /alerts/*  — guardian alert feed
# ---------------------------------------------------------------------------

@app.get("/alerts", response_model=List[AlertOut], tags=["Alerts"])
def get_alerts(
    current_user: User = Depends(_require_auth),
    db: Session = Depends(get_db),
):
    if current_user.role == RoleEnum.guardian:
        blind = db.query(User).filter(User.guardian_id == current_user.id).first()
        if not blind:
            return []
        return db.query(Alert).filter(Alert.blind_id == blind.id).order_by(Alert.created_at.desc()).limit(50).all()

    if current_user.role == RoleEnum.admin:
        return db.query(Alert).order_by(Alert.created_at.desc()).limit(100).all()

    raise HTTPException(status_code=403, detail="Access denied.")


@app.get("/alerts/{blind_id}", response_model=List[AlertOut], tags=["Alerts"])
def get_alerts_for_blind(
    blind_id: int,
    current_user: User = Depends(_require_role(RoleEnum.admin)),
    db: Session = Depends(get_db),
):
    return db.query(Alert).filter(Alert.blind_id == blind_id).order_by(Alert.created_at.desc()).all()


# ---------------------------------------------------------------------------
# /guardian/*  — guardian dashboard
# ---------------------------------------------------------------------------

@app.get("/guardian/dashboard", response_model=GuardianDashboardOut, tags=["Guardian"])
def guardian_dashboard(
    current_user: User = Depends(_require_role(RoleEnum.guardian)),
    db: Session = Depends(get_db),
):
    # Find the blind user linked to this guardian
    blind = db.query(User).filter(User.guardian_id == current_user.id).first()
    recent_alerts = []
    active_session = None

    if blind:
        # Exact query from spec:
        # SELECT * FROM alerts
        # WHERE blind_id IN (SELECT id FROM users WHERE guardian_id = current_guardian_id)
        # ORDER BY created_at DESC LIMIT 10
        blind_ids = [
            row.id for row in
            db.query(User.id).filter(User.guardian_id == current_user.id).all()
        ]
        recent_alerts = (
            db.query(Alert)
            .filter(Alert.blind_id.in_(blind_ids))
            .order_by(Alert.created_at.desc())
            .limit(10)
            .all()
        )
        active_session = (
            db.query(NavSession)
            .filter(
                NavSession.blind_id == blind.id,
                NavSession.status == SessionStatusEnum.active,
            )
            .first()
        )

    return GuardianDashboardOut(
        guardian=current_user,
        blind_user=blind,
        recent_alerts=recent_alerts,
        active_session=active_session,
    )


# ---------------------------------------------------------------------------
# /session/*  — blind user navigation sessions
# ---------------------------------------------------------------------------

@app.post("/session/start", response_model=SessionOut, status_code=201, tags=["Sessions"])
def start_session(
    current_user: User = Depends(_require_role(RoleEnum.blind)),
    db: Session = Depends(get_db),
):
    # End any existing active session first
    existing = db.query(NavSession).filter(
        NavSession.blind_id == current_user.id,
        NavSession.status == SessionStatusEnum.active,
    ).first()
    if existing:
        from datetime import datetime
        existing.status = SessionStatusEnum.ended
        existing.ended_at = datetime.utcnow()

    session = NavSession(blind_id=current_user.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.post("/session/end", response_model=SessionOut, tags=["Sessions"])
def end_session(
    current_user: User = Depends(_require_role(RoleEnum.blind)),
    db: Session = Depends(get_db),
):
    from datetime import datetime
    session = db.query(NavSession).filter(
        NavSession.blind_id == current_user.id,
        NavSession.status == SessionStatusEnum.active,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="No active session found.")
    session.status = SessionStatusEnum.ended
    session.ended_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session


@app.get("/session/active", response_model=Optional[SessionOut], tags=["Sessions"])
def get_active_session(
    current_user: User = Depends(_require_auth),
    db: Session = Depends(get_db),
):
    blind_id = current_user.id if current_user.role == RoleEnum.blind else None
    if current_user.role == RoleEnum.guardian:
        blind = db.query(User).filter(User.guardian_id == current_user.id).first()
        blind_id = blind.id if blind else None
    if not blind_id:
        return None
    return db.query(NavSession).filter(
        NavSession.blind_id == blind_id,
        NavSession.status == SessionStatusEnum.active,
    ).first()


@app.get("/session/history", response_model=List[SessionOut], tags=["Sessions"])
def session_history(
    current_user: User = Depends(_require_role(RoleEnum.admin)),
    db: Session = Depends(get_db),
):
    return db.query(NavSession).order_by(NavSession.started_at.desc()).limit(100).all()


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


# ---------------------------------------------------------------------------
# /guardian/tracking  — real timeline + session stats
# ---------------------------------------------------------------------------

@app.get("/guardian/tracking", tags=["Guardian"])
def guardian_tracking(
    current_user: User = Depends(_require_role(RoleEnum.guardian)),
    db: Session = Depends(get_db),
):
    blind = db.query(User).filter(User.guardian_id == current_user.id).first()
    if not blind:
        return {"timeline": [], "session": {"status": "No blind user", "duration_minutes": 0, "alert_count": 0}, "blind_name": ""}

    alerts = (
        db.query(Alert)
        .filter(Alert.blind_id == blind.id)
        .order_by(Alert.created_at.desc())
        .limit(20)
        .all()
    )

    active_session = (
        db.query(NavSession)
        .filter(NavSession.blind_id == blind.id, NavSession.status == SessionStatusEnum.active)
        .first()
    )

    duration = 0
    if active_session:
        now = datetime.now(timezone.utc)
        started = active_session.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        duration = int((now - started).total_seconds() / 60)

    top_level = "Safe"
    if alerts:
        if alerts[0].level.value == "danger":
            top_level = "Danger"
        elif alerts[0].level.value == "warning":
            top_level = "Warning"

    timeline = [
        {
            "time": a.created_at.strftime("%H:%M"),
            "event": a.message,
            "level": a.level.value,
        }
        for a in alerts
    ]

    return {
        "blind_name": blind.name,
        "blind_phone": blind.phone,
        "is_scanning": active_session is not None,
        "timeline": timeline,
        "session": {
            "status": top_level,
            "duration_minutes": duration,
            "alert_count": len(alerts),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
