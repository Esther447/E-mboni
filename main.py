"""
main.py — E-mboni FastAPI Backend
Database: Firebase Cloud Firestore

Handles:
- Authentication
- User registration/login
- YOLO object detection
- Alerts
- Sessions
- Guardian dashboard
- Guardian tracking
- Emergency
- Admin
- Location/device endpoints
"""

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime, timezone, date

import cv2
import numpy as np
import uvicorn
import logging

from ultralytics import YOLO

import firestore_service as fs

from database import (
    init_db,
    get_db,
    verify_password,
    hash_password,
    RoleEnum,
    AlertLevelEnum,
    SessionStatusEnum,
    LanguageEnum,
    VoiceSpeedEnum,
    StatusEnum,
)

from auth import (
    create_token,
    decode_token,
    get_current_user_from_token,
)

from models import (
    RegisterRequest,
    RegisterResponse,
    UserOut,
    BlindUserOut,
    LoginRequest,
    LoginResponse,
    LoginUserOut,
    BlindUserSummary,
    DetectedObject,
    DetectionResult,
    AlertOut,
    SessionOut,
    UserAdminOut,
    GuardianDashboardOut,
)

from spatial_engine import (
    RawDetection,
    process_detections,
    ALL_OBJECTS,
)

from events import event_store
from memory import CrowdDetector, ConsistencyFilter


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("emboni")


# ---------------------------------------------------------------------------
# AI / Memory
# ---------------------------------------------------------------------------

_crowd_detector = CrowdDetector()
_consistency = ConsistencyFilter()


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

This API powers the E-mboni mobile app.

Features:
- Authentication
- Guardian + blind user registration
- Login
- YOLO object detection
- Safety alerts
- Navigation sessions
- Guardian dashboard
- Guardian tracking
- Emergency
- Admin management
""",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# YOLO models
# ---------------------------------------------------------------------------

custom_model = YOLO(
    "yolov8n.onnx",
    task="detect",
)

general_model = YOLO(
    "yolov8n.pt",
    task="detect",
)


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

def _require_auth(
    authorization: str = Header(...),
):
    """
    Validate Bearer token and return current user.
    """

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header.",
        )

    token = authorization.split(" ", 1)[1].strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token.",
        )

    return get_current_user_from_token(token)


def _require_role(role: RoleEnum):
    """
    Require a specific user role.
    """

    def checker(
        current_user: dict = Depends(_require_auth),
    ):
        if current_user["role"] != role.value:
            raise HTTPException(
                status_code=403,
                detail="Access denied.",
            )

        return current_user

    return checker


# ---------------------------------------------------------------------------
# Response conversion helpers
# ---------------------------------------------------------------------------

def _user_to_admin_out(u: dict) -> UserAdminOut:
    """
    Convert Firestore user dictionary to UserAdminOut.
    """

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
    """
    Convert Firestore alert dictionary to AlertOut.
    """

    return AlertOut(
        id=a["id"],
        blind_id=a["blind_id"],
        message=a["message"],
        level=a["level"],
        is_read=a.get("is_read", False),
        created_at=a["created_at"],
    )


def _session_to_out(s: dict) -> SessionOut:
    """
    Convert Firestore session dictionary to SessionOut.
    """

    return SessionOut(
        id=s["id"],
        blind_id=s["blind_id"],
        started_at=s["started_at"],
        ended_at=s.get("ended_at"),
        status=s["status"],
    )


# ===========================================================================
# AUTHENTICATION
# ===========================================================================

# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------

@app.post(
    "/auth/register",
    response_model=RegisterResponse,
    status_code=201,
    tags=["Authentication"],
)
async def register(
    body: RegisterRequest,
    db=Depends(get_db),
):
    """
    Register a guardian and blind user together.
    """

    import asyncio
    from functools import partial

    # Check both phone numbers
    for phone in [
        body.guardian.phone,
        body.blind_user.phone,
    ]:
        if fs.get_user_by_phone(phone):
            raise HTTPException(
                status_code=422,
                detail="Phone number already registered",
            )

    # Hash guardian password
    loop = asyncio.get_event_loop()

    guardian_hash = await loop.run_in_executor(
        None,
        partial(
            hash_password,
            body.guardian.password,
        ),
    )

    # Blind user's initial password
    blind_password = body.blind_user.phone[-6:]

    blind_hash = await loop.run_in_executor(
        None,
        partial(
            hash_password,
            blind_password,
        ),
    )

    # Create guardian
    guardian = fs.create_user(
        name=body.guardian.name,
        phone=body.guardian.phone,
        password_hash=guardian_hash,
        role=RoleEnum.guardian.value,
        relationship=body.guardian.relationship,
    )

    # Create blind user
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

    logger.info(
        f"REGISTER | guardian={guardian['phone']} "
        f"blind={blind_user['phone']}"
    )

    # Create guardian token
    token = create_token(
        guardian["id"],
        guardian["role"],
    )

    return RegisterResponse(
        guardian=UserOut(
            id=guardian["id"],
            name=guardian["name"],
            phone=guardian["phone"],
            role=guardian["role"],
        ),
        blind_user=BlindUserOut(
            id=blind_user["id"],
            name=blind_user["name"],
            phone=blind_user["phone"],
            role=blind_user["role"],
            guardian_id=guardian["id"],
        ),
        token=token,
    )


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

@app.post(
    "/auth/login",
    response_model=LoginResponse,
    tags=["Authentication"],
)
def login(
    body: LoginRequest,
    db=Depends(get_db),
):
    """
    Login using phone and password.
    """

    user = fs.get_user_by_phone(body.phone)

    if not user:
        logger.warning(
            f"LOGIN FAILED | phone={body.phone}"
        )

        raise HTTPException(
            status_code=401,
            detail="Wrong phone or password.",
        )

    if not verify_password(
        body.password,
        user["password_hash"],
    ):
        logger.warning(
            f"LOGIN FAILED | phone={body.phone}"
        )

        raise HTTPException(
            status_code=401,
            detail="Wrong phone or password.",
        )

    logger.info(
        f"LOGIN OK | phone={user['phone']} "
        f"role={user['role']}"
    )

    blind_summary = None

    if user["role"] == RoleEnum.guardian.value:

        linked_list = fs.get_blind_users_for_guardian(
            user["id"]
        )

        if linked_list:

            linked = linked_list[0]

            blind_summary = BlindUserSummary(
                id=linked["id"],
                name=linked["name"],
                status=linked.get(
                    "status",
                    "active",
                ),
                language=linked.get(
                    "language",
                    "en",
                ),
            )

    return LoginResponse(
        token=create_token(
            user["id"],
            user["role"],
        ),
        user=LoginUserOut(
            id=user["id"],
            name=user["name"],
            role=user["role"],
            language=user.get(
                "language",
                "en",
            ),
            voice_speed=user.get(
                "voice_speed",
                "Normal",
            ),
            blind_user=blind_summary,
        ),
    )


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------

@app.get(
    "/auth/me",
    tags=["Authentication"],
)
def get_me(
    current_user: dict = Depends(_require_auth),
):
    return {
        "id": current_user["id"],
        "name": current_user["name"],
        "phone": current_user["phone"],
        "role": current_user["role"],
        "language": current_user.get(
            "language",
            "en",
        ),
        "voice_speed": current_user.get(
            "voice_speed",
            "Normal",
        ),
        "status": current_user.get(
            "status",
            "active",
        ),
        "emergency_phone": current_user.get(
            "emergency_phone"
        ),
        "guardian_id": current_user.get(
            "guardian_id"
        ),
    }


# ===========================================================================
# DETECTION
# ===========================================================================

MOVING_OBJECTS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
    "dog",
    "cat",
    "vehicle",
    "van",
    "scooter",
    "animal",
}


HIGH_DANGER_OBJECTS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "vehicle",
    "van",
}


MEDIUM_DANGER_OBJECTS = {
    "bicycle",
    "scooter",
    "person",
    "dog",
    "cat",
    "animal",
    "chair",
    "table",
    "bench",
    "pole",
    "fire hydrant",
    "trash can",
    "staircase",
    "stairs",
    "step",
}


CRITICAL_CLOSE_OBJECTS = {
    "car",
    "stair",
    "truck",
    "bus",
    "motorcycle",
    "staircase",
    "stairs",
}


DIRECTION_MAP = {
    "to your far left": "left",
    "on your left": "left",
    "straight ahead": "center",
    "on your right": "right",
    "to your far right": "right",
}


def _norm_area_to_meters(
    norm_area: float,
) -> float:

    if norm_area > 0.4:
        return 0.5

    if norm_area > 0.2:
        return 1.0

    if norm_area > 0.1:
        return 2.0

    if norm_area > 0.05:
        return 3.5

    if norm_area > 0.02:
        return 6.0

    if norm_area > 0.01:
        return 10.0

    return 15.0


def _danger_level(
    name: str,
    distance: float,
) -> str:

    if (
        name in CRITICAL_CLOSE_OBJECTS
        and distance <= 1.5
    ):
        return "danger"

    if name in HIGH_DANGER_OBJECTS:

        if distance <= 5:
            return "danger"

        return "warning"

    if name in MEDIUM_DANGER_OBJECTS:

        if distance <= 1.5:
            return "danger"

        if distance <= 4:
            return "warning"

        return "safe"

    if distance <= 1:
        return "warning"

    return "safe"


def _map_direction(
    direction_str: str,
) -> str:

    base = direction_str.split(",")[0].strip()

    return DIRECTION_MAP.get(
        base,
        "center",
    )


def _build_summary(
    objects: list,
) -> str:

    if not objects:
        return "Path clear."

    top = objects[0]

    moving = (
        "moving "
        if top.isMoving
        else ""
    )

    direction_text = (
        "ahead"
        if top.direction == "center"
        else top.direction
    )

    return (
        f"{moving}"
        f"{top.name}, "
        f"{top.distanceMeters:.1f} meters "
        f"{direction_text}"
    )


PATH_CLEAR_SILENCE_SECONDS = 5.0


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------

@app.post(
    "/detect",
    response_model=DetectionResult,
    tags=["Detection"],
)
async def detect(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    db=Depends(get_db),
):
    """
    Receive camera frame and run YOLO detection.
    """

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty image file received.",
        )

    nparr = np.frombuffer(
        contents,
        np.uint8,
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR,
    )

    if img is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode image. "
                "Send a valid JPEG."
            ),
        )

    h, w = img.shape[:2]

    # ---------------------------------------------------------------
    # Identify blind user from token
    # ---------------------------------------------------------------

    blind_user: Optional[dict] = None

    if (
        authorization
        and authorization.startswith("Bearer ")
    ):

        try:

            token = authorization.split(
                " ",
                1,
            )[1]

            payload = decode_token(token)

            user_id = int(
                payload["sub"]
            )

            user = fs.get_user_by_id(
                user_id
            )

            if (
                user
                and user["role"]
                == RoleEnum.blind.value
            ):
                blind_user = user

        except Exception as exc:

            logger.warning(
                f"DETECT AUTH WARNING | {exc}"
            )

    # ---------------------------------------------------------------
    # YOLO inference
    # ---------------------------------------------------------------

    results_custom = custom_model(
        img,
        conf=0.6,
    )

    results_general = general_model(
        img,
        conf=0.6,
    )

    raw_detections: list[
        RawDetection
    ] = []

    seen_labels: set[str] = set()

    for result in (
        results_custom + results_general
    ):

        for box in result.boxes:

            label = result.names[
                int(box.cls[0])
            ]

            if label not in ALL_OBJECTS:
                continue

            if label in seen_labels:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            raw_detections.append(
                RawDetection(
                    label=label,
                    box_area=float(
                        (x2 - x1)
                        * (y2 - y1)
                    ),
                    norm_x=float(
                        (
                            (x1 + x2) / 2
                        ) / w
                    ),
                    norm_y=float(
                        (
                            (y1 + y2) / 2
                        ) / h
                    ),
                    box_w=float(
                        x2 - x1
                    ),
                    box_h=float(
                        y2 - y1
                    ),
                )
            )

            seen_labels.add(label)

    # ---------------------------------------------------------------
    # Spatial processing
    # ---------------------------------------------------------------

    spatial_results = process_detections(
        raw_detections
    )

    label_priorities = {
        item.object: item.priority
        for item in spatial_results
    }

    _consistency.update(
        [
            detection.label
            for detection in raw_detections
        ],
        label_priorities,
    )

    raw_detections = [
        detection
        for detection in raw_detections
        if _consistency.is_confirmed(
            detection.label
        )
    ]

    spatial_results = [
        spatial
        for spatial in spatial_results
        if _consistency.is_confirmed(
            spatial.object
        )
    ]

    # ---------------------------------------------------------------
    # Crowd detection
    # ---------------------------------------------------------------

    is_crowd, crowd_message = (
        _crowd_detector.evaluate(
            raw_detections,
            spatial_results,
        )
    )

    # ---------------------------------------------------------------
    # Build detected objects
    # ---------------------------------------------------------------

    raw_by_label = {
        detection.label: detection
        for detection in raw_detections
    }

    detected_objects: list[
        DetectedObject
    ] = []

    for spatial in spatial_results:

        raw = raw_by_label.get(
            spatial.object
        )

        if not raw:
            continue

        norm_area = (
            raw.box_area
            / (w * h)
        )

        distance_m = (
            _norm_area_to_meters(
                norm_area
            )
        )

        direction = _map_direction(
            spatial.direction
        )

        detected_objects.append(
            DetectedObject(
                name=raw.label,
                isMoving=(
                    raw.label
                    in MOVING_OBJECTS
                ),
                distanceMeters=distance_m,
                direction=direction,
                dangerLevel=_danger_level(
                    raw.label,
                    distance_m,
                ),
            )
        )

    # ---------------------------------------------------------------
    # Sort by danger
    # ---------------------------------------------------------------

    danger_order = {
        "danger": 0,
        "warning": 1,
        "safe": 2,
    }

    detected_objects.sort(
        key=lambda obj:
        danger_order.get(
            obj.dangerLevel,
            2,
        )
    )

    top_danger = (
        detected_objects[0].dangerLevel
        if detected_objects
        else "safe"
    )

    # ---------------------------------------------------------------
    # Create alert for blind user
    # ---------------------------------------------------------------

    if (
        blind_user
        and detected_objects
    ):

        top = detected_objects[0]

        if top.dangerLevel in (
            "danger",
            "warning",
        ):

            event_store.log_safety_event(
                hazard=top.name,
                direction=top.direction,
                distance=str(
                    top.distanceMeters
                ),
            )

            top_spatial = next(
                (
                    spatial
                    for spatial
                    in spatial_results
                    if spatial.object
                    == top.name
                ),
                None,
            )

            fs.create_alert(
                blind_id=blind_user["id"],
                message=(
                    top_spatial.speech
                    if top_spatial
                    else top.name
                ),
                level=top.dangerLevel,
            )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    if is_crowd and crowd_message:

        summary = crowd_message

    elif not detected_objects:

        if blind_user:

            last = fs.get_last_alert_for_blind(
                blind_user["id"]
            )

            if last:

                now = datetime.now(
                    timezone.utc
                )

                last_time = last[
                    "created_at"
                ]

                if last_time.tzinfo is None:
                    last_time = (
                        last_time.replace(
                            tzinfo=timezone.utc
                        )
                    )

                seconds_ago = (
                    now - last_time
                ).total_seconds()

                if (
                    seconds_ago
                    >= PATH_CLEAR_SILENCE_SECONDS
                ):
                    summary = "Path clear."
                else:
                    summary = ""

            else:
                summary = "Path clear."

        else:
            summary = "Path clear."

    else:

        summary = _build_summary(
            detected_objects
        )

    logger.info(
        "DETECT | "
        f"user={blind_user['phone'] if blind_user else 'anonymous'} | "
        f"topDanger={top_danger} | "
        f"objects={len(detected_objects)}"
    )

    return DetectionResult(
        objects=detected_objects,
        summary=summary,
        topDanger=top_danger,
    )


# ===========================================================================
# ADMIN USERS
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /users
# ---------------------------------------------------------------------------

@app.get(
    "/users",
    response_model=List[UserAdminOut],
    tags=["Admin"],
)
def list_users(
    current_user: dict = Depends(
        _require_role(RoleEnum.admin)
    ),
    db=Depends(get_db),
):
    return [
        _user_to_admin_out(user)
        for user in fs.get_all_users()
    ]


# ---------------------------------------------------------------------------
# GET /users/{user_id}
# ---------------------------------------------------------------------------

@app.get(
    "/users/{user_id}",
    response_model=UserAdminOut,
    tags=["Admin"],
)
def get_user(
    user_id: int,
    current_user: dict = Depends(
        _require_role(RoleEnum.admin)
    ),
    db=Depends(get_db),
):
    user = fs.get_user_by_id(
        user_id
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found.",
        )

    return _user_to_admin_out(user)


# ---------------------------------------------------------------------------
# PATCH /users/{user_id}/status
# ---------------------------------------------------------------------------

@app.patch(
    "/users/{user_id}/status",
    tags=["Admin"],
)
def update_user_status(
    user_id: int,
    status: str,
    current_user: dict = Depends(
        _require_role(RoleEnum.admin)
    ),
    db=Depends(get_db),
):
    valid_statuses = [
        status_item.value
        for status_item in StatusEnum
    ]

    if status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail="Invalid status value.",
        )

    user = fs.update_user_status(
        user_id,
        status,
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found.",
        )

    return {
        "id": user_id,
        "status": status,
    }


# ===========================================================================
# ALERTS
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /alerts
# ---------------------------------------------------------------------------

@app.get(
    "/alerts",
    response_model=List[AlertOut],
    tags=["Alerts"],
)
def get_alerts(
    current_user: dict = Depends(_require_auth),
    db=Depends(get_db),
):

    # Guardian
    if (
        current_user["role"]
        == RoleEnum.guardian.value
    ):

        blind_list = (
            fs.get_blind_users_for_guardian(
                current_user["id"]
            )
        )

        if not blind_list:
            return []

        return [
            _alert_to_out(alert)
            for alert in fs.get_alerts_for_blind(
                blind_list[0]["id"],
                limit=50,
            )
        ]

    # Admin
    if (
        current_user["role"]
        == RoleEnum.admin.value
    ):

        return [
            _alert_to_out(alert)
            for alert in fs.get_all_alerts(
                limit=100
            )
        ]

    raise HTTPException(
        status_code=403,
        detail="Access denied.",
    )


# ---------------------------------------------------------------------------
# GET /alerts/{blind_id}
# ---------------------------------------------------------------------------

@app.get(
    "/alerts/{blind_id}",
    response_model=List[AlertOut],
    tags=["Alerts"],
)
def get_alerts_for_blind(
    blind_id: int,
    current_user: dict = Depends(
        _require_role(RoleEnum.admin)
    ),
    db=Depends(get_db),
):
    return [
        _alert_to_out(alert)
        for alert in fs.get_alerts_for_blind(
            blind_id
        )
    ]


# ===========================================================================
# GUARDIAN
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /guardian/dashboard
# ---------------------------------------------------------------------------

@app.get(
    "/guardian/dashboard",
    response_model=GuardianDashboardOut,
    tags=["Guardian"],
)
def guardian_dashboard(
    current_user: dict = Depends(
        _require_role(RoleEnum.guardian)
    ),
    db=Depends(get_db),
):
    """
    Guardian dashboard.

    FIXED:
    The previous code called:


    That function does not exist.

    The Firestore service provides:

        fs.get_alerts_for_blinds()

    so this endpoint now uses that function.
    """

    # Get blind users linked to guardian
    blind_list = (
        fs.get_blind_users_for_guardian(
            current_user["id"]
        )
    )

    recent_alerts = []
    active_session = None

    blind = (
        blind_list[0]
        if blind_list
        else None
    )

    if blind_list:

        # -----------------------------------------------------------
        # Get alerts
        #
        # IMPORTANT:
        # Use the existing Firestore function.
        # -----------------------------------------------------------

        blind_ids = [
            user["id"]
            for user in blind_list
        ]

        recent_alerts_raw = (
            fs.get_alerts_for_blinds(
                blind_ids,
                limit=10,
            )
        )

        recent_alerts = [
            _alert_to_out(alert)
            for alert in recent_alerts_raw
        ]

        # -----------------------------------------------------------
        # Active session
        # -----------------------------------------------------------

        session = fs.get_active_session(
            blind["id"]
        )

        if session:
            active_session = (
                _session_to_out(session)
            )

    return GuardianDashboardOut(
        guardian=_user_to_admin_out(
            current_user
        ),
        blind_user=(
            _user_to_admin_out(blind)
            if blind
            else None
        ),
        recent_alerts=recent_alerts,
        active_session=active_session,
    )


# ---------------------------------------------------------------------------
# GET /guardian/tracking
# ---------------------------------------------------------------------------

@app.get(
    "/guardian/tracking",
    tags=["Guardian"],
)
def guardian_tracking(
    current_user: dict = Depends(
        _require_role(RoleEnum.guardian)
    ),
    db=Depends(get_db),
):

    blind_list = (
        fs.get_blind_users_for_guardian(
            current_user["id"]
        )
    )

    if not blind_list:

        return {
            "timeline": [],
            "session": {
                "status": "No blind user",
                "duration_minutes": 0,
                "alert_count": 0,
            },
            "blind_name": "",
        }

    blind = blind_list[0]

    alerts = fs.get_alerts_for_blind(
        blind["id"],
        limit=20,
    )

    session = fs.get_active_session(
        blind["id"]
    )

    # ---------------------------------------------------------------
    # Session duration
    # ---------------------------------------------------------------

    duration = 0

    if session:

        now = datetime.now(
            timezone.utc
        )

        started = session[
            "started_at"
        ]

        if started.tzinfo is None:
            started = started.replace(
                tzinfo=timezone.utc
            )

        duration = int(
            (
                now - started
            ).total_seconds()
            / 60
        )

    # ---------------------------------------------------------------
    # Current safety level
    # ---------------------------------------------------------------

    top_level = "Safe"

    if alerts:

        if alerts[0]["level"] == "danger":
            top_level = "Danger"

        elif alerts[0]["level"] == "warning":
            top_level = "Warning"

    # ---------------------------------------------------------------
    # Timeline
    # ---------------------------------------------------------------

    timeline = []

    for alert in alerts:

        created_at = alert[
            "created_at"
        ]

        if hasattr(
            created_at,
            "strftime",
        ):
            time_value = created_at.strftime(
                "%H:%M"
            )
        else:
            time_value = str(
                created_at
            )

        timeline.append(
            {
                "time": time_value,
                "event": alert["message"],
                "level": alert["level"],
            }
        )

    return {
        "blind_name": blind["name"],
        "blind_phone": blind["phone"],
        "is_scanning": session is not None,
        "timeline": timeline,
        "session": {
            "status": top_level,
            "duration_minutes": duration,
            "alert_count": len(alerts),
        },
    }


# ---------------------------------------------------------------------------
# GET /guardian/alerts
# ---------------------------------------------------------------------------

@app.get(
    "/guardian/alerts",
    response_model=List[AlertOut],
    tags=["Guardian"],
)
def guardian_alerts(
    current_user: dict = Depends(
        _require_role(RoleEnum.guardian)
    ),
    db=Depends(get_db),
):

    blind_list = (
        fs.get_blind_users_for_guardian(
            current_user["id"]
        )
    )

    if not blind_list:
        return []

    alerts = fs.get_alerts_for_blind(
        blind_list[0]["id"],
        limit=50,
    )

    return [
        _alert_to_out(alert)
        for alert in alerts
    ]


# ---------------------------------------------------------------------------
# GET /guardian/emergency
# ---------------------------------------------------------------------------

@app.get(
    "/guardian/emergency",
    tags=["Guardian"],
)
def guardian_emergency_get(
    current_user: dict = Depends(
        _require_role(RoleEnum.guardian)
    ),
    db=Depends(get_db),
):

    blind_list = (
        fs.get_blind_users_for_guardian(
            current_user["id"]
        )
    )

    if not blind_list:
        raise HTTPException(
            status_code=404,
            detail="No blind user linked.",
        )

    blind = blind_list[0]

    return {
        "blind_name": blind["name"],
        "emergency_phone": blind.get(
            "emergency_phone"
        ),
        "guardian_phone": current_user[
            "phone"
        ],
    }


# ---------------------------------------------------------------------------
# POST /guardian/emergency
# ---------------------------------------------------------------------------

@app.post(
    "/guardian/emergency",
    tags=["Guardian"],
)
def guardian_emergency_post(
    current_user: dict = Depends(
        _require_role(RoleEnum.guardian)
    ),
    db=Depends(get_db),
):
    """
    Trigger emergency.

    The Flutter app uses the returned phone
    number to initiate the actual phone call.
    """

    blind_list = (
        fs.get_blind_users_for_guardian(
            current_user["id"]
        )
    )

    if not blind_list:
        raise HTTPException(
            status_code=404,
            detail="No blind user linked.",
        )

    blind = blind_list[0]

    logger.info(
        f"EMERGENCY | "
        f"guardian={current_user['phone']} "
        f"blind={blind['phone']}"
    )

    return {
        "blind_name": blind["name"],
        "emergency_phone": blind.get(
            "emergency_phone"
        ),
        "guardian_phone": current_user[
            "phone"
        ],
        "triggered": True,
    }


# ---------------------------------------------------------------------------
# POST /guardian/alerts/{alert_id}/read
# ---------------------------------------------------------------------------

@app.post(
    "/guardian/alerts/{alert_id}/read",
    tags=["Guardian"],
)
def mark_alert_read(
    alert_id: int,
    current_user: dict = Depends(
        _require_role(RoleEnum.guardian)
    ),
    db=Depends(get_db),
):
    """
    Mark an alert as read.

    Guardian can only mark alerts belonging
    to their linked blind user.
    """

    alert = fs.get_alert_by_id(
        alert_id
    )

    if not alert:
        raise HTTPException(
            status_code=404,
            detail="Alert not found.",
        )

    blind_list = (
        fs.get_blind_users_for_guardian(
            current_user["id"]
        )
    )

    blind_ids = [
        user["id"]
        for user in blind_list
    ]

    if alert["blind_id"] not in blind_ids:
        raise HTTPException(
            status_code=403,
            detail="Access denied.",
        )

    updated = fs.mark_alert_read(
        alert_id
    )

    return _alert_to_out(
        updated
    )


# ===========================================================================
# SESSIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# POST /session/start
# ---------------------------------------------------------------------------

@app.post(
    "/session/start",
    response_model=SessionOut,
    status_code=201,
    tags=["Sessions"],
)
def start_session(
    current_user: dict = Depends(
        _require_role(RoleEnum.blind)
    ),
    db=Depends(get_db),
):

    session = fs.start_session(
        current_user["id"]
    )

    return _session_to_out(
        session
    )


# ---------------------------------------------------------------------------
# POST /session/end
# ---------------------------------------------------------------------------

@app.post(
    "/session/end",
    response_model=SessionOut,
    tags=["Sessions"],
)
def end_session(
    current_user: dict = Depends(
        _require_role(RoleEnum.blind)
    ),
    db=Depends(get_db),
):

    session = fs.end_session(
        current_user["id"]
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="No active session found.",
        )

    return _session_to_out(
        session
    )


# ---------------------------------------------------------------------------
# GET /session/active
# ---------------------------------------------------------------------------

@app.get(
    "/session/active",
    response_model=Optional[SessionOut],
    tags=["Sessions"],
)
def get_active_session(
    current_user: dict = Depends(
        _require_auth
    ),
    db=Depends(get_db),
):

    blind_id = None

    # Blind user
    if (
        current_user["role"]
        == RoleEnum.blind.value
    ):

        blind_id = current_user["id"]

    # Guardian
    elif (
        current_user["role"]
        == RoleEnum.guardian.value
    ):

        blind_list = (
            fs.get_blind_users_for_guardian(
                current_user["id"]
            )
        )

        if blind_list:
            blind_id = blind_list[0]["id"]

    if not blind_id:
        return None

    session = fs.get_active_session(
        blind_id
    )

    if not session:
        return None

    return _session_to_out(
        session
    )


# ---------------------------------------------------------------------------
# GET /session/history
# ---------------------------------------------------------------------------

@app.get(
    "/session/history",
    response_model=List[SessionOut],
    tags=["Sessions"],
)
def session_history(
    current_user: dict = Depends(
        _require_role(RoleEnum.admin)
    ),
    db=Depends(get_db),
):

    return [
        _session_to_out(session)
        for session in fs.get_all_sessions(
            limit=100
        )
    ]


# ===========================================================================
# ADMIN
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /admin/overview
# ---------------------------------------------------------------------------

@app.get(
    "/admin/overview",
    tags=["Admin"],
)
def admin_overview(
    current_user: dict = Depends(
        _require_role(RoleEnum.admin)
    ),
    db=Depends(get_db),
):

    all_users = fs.get_all_users()

    total_users = len(
        all_users
    )

    total_guardians = sum(
        1
        for user in all_users
        if user["role"]
        == RoleEnum.guardian.value
    )

    active_blind_ids = (
        fs.get_active_session_ids()
    )

    active_now = len(
        active_blind_ids
    )

    today_start = (
        datetime.combine(
            date.today(),
            datetime.min.time(),
        ).replace(
            tzinfo=timezone.utc
        )
    )

    alerts_today = (
        fs.count_alerts_since(
            today_start
        )
    )

    active_users = []

    for blind_id in active_blind_ids:

        user = fs.get_user_by_id(
            blind_id
        )

        if user:

            active_users.append(
                {
                    "id": user["id"],
                    "name": user["name"],
                    "status": "Scanning",
                }
            )

    recent_alerts_raw = (
        fs.get_alerts_with_user(
            limit=10
        )
    )

    recent_alerts = []

    for alert in recent_alerts_raw:

        created_at = alert[
            "created_at"
        ]

        if hasattr(
            created_at,
            "isoformat",
        ):
            created_at_value = (
                created_at.isoformat()
            )
        else:
            created_at_value = str(
                created_at
            )

        recent_alerts.append(
            {
                "id": alert["id"],
                "user_name": alert.get(
                    "user_name",
                    "Unknown",
                ),
                "message": alert[
                    "message"
                ],
                "level": alert[
                    "level"
                ],
                "created_at": created_at_value,
            }
        )

    return {
        "total_users": total_users,
        "total_guardians": total_guardians,
        "active_now": active_now,
        "alerts_today": alerts_today,
        "active_users": active_users,
        "recent_alerts": recent_alerts,
    }


# ---------------------------------------------------------------------------
# GET /admin/logs
# ---------------------------------------------------------------------------

@app.get(
    "/admin/logs",
    tags=["Admin"],
)
def admin_logs(
    current_user: dict = Depends(
        _require_role(RoleEnum.admin)
    ),
    db=Depends(get_db),
):

    rows = fs.get_alerts_with_user(
        limit=200
    )

    response = []

    for alert in rows:

        created_at = alert[
            "created_at"
        ]

        if hasattr(
            created_at,
            "isoformat",
        ):
            created_at_value = (
                created_at.isoformat()
            )
        else:
            created_at_value = str(
                created_at
            )

        response.append(
            {
                "id": alert["id"],
                "blind_id": alert[
                    "blind_id"
                ],
                "user_name": alert.get(
                    "user_name",
                    "Unknown",
                ),
                "message": alert[
                    "message"
                ],
                "level": alert[
                    "level"
                ],
                "created_at": created_at_value,
            }
        )

    return response


# ===========================================================================
# LEGACY LOCATION / DEVICE ENDPOINTS
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /location/last-known
# ---------------------------------------------------------------------------

@app.get(
    "/location/last-known"
)
async def last_known_location(
    user_id: str,
    authorization: Optional[str] = Header(None),
):

    location = (
        event_store.get_last_known_location(
            user_id
        )
    )

    return {
        "location": location,
        "sharing_enabled": (
            location is not None
        ),
    }


# ---------------------------------------------------------------------------
# POST /location/share
# ---------------------------------------------------------------------------

@app.post(
    "/location/share"
)
async def toggle_location_sharing(
    user_id: str,
    enabled: bool,
):

    event_store.enable_location_sharing(
        user_id,
        enabled,
    )

    return {
        "sharing_enabled": enabled
    }


# ---------------------------------------------------------------------------
# GET /devices/status
# ---------------------------------------------------------------------------

@app.get(
    "/devices/status"
)
async def device_status(
    device_id: str,
):

    return {
        "device_id": device_id,
        "status": event_store.get_device_status(
            device_id
        ),
    }


# ---------------------------------------------------------------------------
# POST /devices/ping
# ---------------------------------------------------------------------------

@app.post(
    "/devices/ping"
)
async def device_ping(
    device_id: str,
):

    event_store.ping(
        device_id
    )

    return {
        "status": "ok"
    }


# ===========================================================================
# RUN SERVER
# ===========================================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )