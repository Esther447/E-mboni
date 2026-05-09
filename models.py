"""
models.py — Pydantic request/response schemas matching the frontend contract in Esther.md.
"""

from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class GuardianRegisterIn(BaseModel):
    name: str
    phone: str
    password: str
    relationship: str

class BlindRegisterIn(BaseModel):
    name: str
    phone: str
    emergency_phone: Optional[str] = None
    language: Literal["en", "rw"] = "en"
    voice_speed: Literal["Slow", "Normal", "Fast"] = "Normal"

class RegisterRequest(BaseModel):
    guardian: GuardianRegisterIn
    blind_user: BlindRegisterIn

class UserOut(BaseModel):
    id: int
    name: str
    phone: str
    role: str

class BlindUserOut(BaseModel):
    id: int
    name: str
    phone: str
    role: str
    guardian_id: int

class RegisterResponse(BaseModel):
    guardian: UserOut
    blind_user: BlindUserOut
    token: str

class LoginRequest(BaseModel):
    phone: str
    password: str

class BlindUserSummary(BaseModel):
    id: int
    name: str
    status: str
    language: str

class LoginUserOut(BaseModel):
    id: int
    name: str
    role: str
    language: str
    voice_speed: str
    blind_user: Optional[BlindUserSummary] = None

class LoginResponse(BaseModel):
    token: str
    user: LoginUserOut


# ---------------------------------------------------------------------------
# Detection — frontend format from Esther.md
# ---------------------------------------------------------------------------

class DetectedObject(BaseModel):
    name: str
    isMoving: bool
    distanceMeters: float
    direction: Literal["left", "center", "right"]
    dangerLevel: Literal["safe", "warning", "danger"]

class DetectionResult(BaseModel):
    objects: List[DetectedObject]
    summary: str
    topDanger: Literal["safe", "warning", "danger"]


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class AlertOut(BaseModel):
    id: int
    blind_id: int
    message: str
    level: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionOut(BaseModel):
    id: int
    blind_id: int
    started_at: datetime
    ended_at: Optional[datetime]
    status: str

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Users (admin)
# ---------------------------------------------------------------------------

class UserAdminOut(BaseModel):
    id: int
    name: str
    phone: str
    role: str
    language: str
    voice_speed: str
    status: str
    guardian_id: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Guardian dashboard
# ---------------------------------------------------------------------------

class GuardianDashboardOut(BaseModel):
    guardian: UserAdminOut
    blind_user: Optional[UserAdminOut]
    recent_alerts: List[AlertOut]
    active_session: Optional[SessionOut]
