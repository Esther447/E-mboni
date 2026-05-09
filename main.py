"""
main.py — E-mboni FastAPI Backend
Implements the full contract from Esther.md:
  /auth/register  /auth/login
  /detect         (updated to frontend DetectionResult format)
  /users/*        (admin)
  /alerts/*       (guardian alert feed)
  /guardian/*     (guardian dashboard)
  /session/*      (blind user navigation sessions)
"""

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional, List
import cv2
import numpy as np
import base64
import uvicorn

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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="E-mboni API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

custom_model = YOLO("yolov8n.onnx")
general_model = YOLO("yolov8n.pt")


@app.on_event("startup")
def startup():
    init_db()


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

@app.post("/auth/register", response_model=RegisterResponse, status_code=201)
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
        password_hash=hash_password(body.blind_user.phone[-6:]),  # temp password = last 6 digits
        role=RoleEnum.blind,
        language=body.blind_user.language,
        voice_speed=body.blind_user.voice_speed,
        emergency_phone=body.blind_user.emergency_phone,
        guardian_id=guardian.id,
    )
    db.add(blind_user)
    db.commit()
    db.refresh(guardian)
    db.refresh(blind_user)

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

@app.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == body.phone).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Wrong phone or password.")

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


@app.post("/detect", response_model=DetectionResult)
async def detect(
    file: UploadFile = File(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    # Accept both file upload and base64 body (frontend sends base64)
    if file:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        raise HTTPException(status_code=400, detail="No image provided.")

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    h, w = img.shape[:2]

    results_c = custom_model(img, conf=0.2)
    results_g = general_model(img, conf=0.25)

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
            ))
            seen_labels.add(label)

    spatial_results = process_detections(raw_detections)

    # Log HIGH priority to event store
    for r in spatial_results:
        if r.priority == "HIGH":
            event_store.log_safety_event(hazard=r.object, direction=r.direction, distance=r.distance)

            # Persist alert to DB if we can identify the blind user from token
            if authorization and authorization.startswith("Bearer "):
                try:
                    payload = decode_token(authorization.split(" ", 1)[1])
                    user = db.query(User).filter(User.id == int(payload["sub"])).first()
                    if user and user.role == RoleEnum.blind:
                        level = "danger" if r.priority == "HIGH" else "warning"
                        db.add(Alert(
                            blind_id=user.id,
                            message=r.speech,
                            level=level,
                        ))
                        db.commit()
                except Exception:
                    pass
            break

    # Build frontend-format objects
    detected_objects: list[DetectedObject] = []
    for raw, spatial in zip(raw_detections, spatial_results):
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

    # Sort by danger level
    danger_order = {"danger": 0, "warning": 1, "safe": 2}
    detected_objects.sort(key=lambda o: danger_order[o.dangerLevel])

    top_danger = detected_objects[0].dangerLevel if detected_objects else "safe"
    summary = _build_summary(detected_objects)

    return DetectionResult(objects=detected_objects, summary=summary, topDanger=top_danger)


# ---------------------------------------------------------------------------
# /users/*  — admin only
# ---------------------------------------------------------------------------

@app.get("/users", response_model=List[UserAdminOut])
def list_users(
    current_user: User = Depends(_require_role(RoleEnum.admin)),
    db: Session = Depends(get_db),
):
    return db.query(User).all()


@app.get("/users/{user_id}", response_model=UserAdminOut)
def get_user(
    user_id: int,
    current_user: User = Depends(_require_role(RoleEnum.admin)),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


@app.patch("/users/{user_id}/status")
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

@app.get("/alerts", response_model=List[AlertOut])
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


@app.get("/alerts/{blind_id}", response_model=List[AlertOut])
def get_alerts_for_blind(
    blind_id: int,
    current_user: User = Depends(_require_role(RoleEnum.admin)),
    db: Session = Depends(get_db),
):
    return db.query(Alert).filter(Alert.blind_id == blind_id).order_by(Alert.created_at.desc()).all()


# ---------------------------------------------------------------------------
# /guardian/*  — guardian dashboard
# ---------------------------------------------------------------------------

@app.get("/guardian/dashboard", response_model=GuardianDashboardOut)
def guardian_dashboard(
    current_user: User = Depends(_require_role(RoleEnum.guardian)),
    db: Session = Depends(get_db),
):
    blind = db.query(User).filter(User.guardian_id == current_user.id).first()
    recent_alerts = []
    active_session = None

    if blind:
        recent_alerts = (
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

    return GuardianDashboardOut(
        guardian=current_user,
        blind_user=blind,
        recent_alerts=recent_alerts,
        active_session=active_session,
    )


# ---------------------------------------------------------------------------
# /session/*  — blind user navigation sessions
# ---------------------------------------------------------------------------

@app.post("/session/start", response_model=SessionOut, status_code=201)
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


@app.post("/session/end", response_model=SessionOut)
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


@app.get("/session/active", response_model=Optional[SessionOut])
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


@app.get("/session/history", response_model=List[SessionOut])
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
