"""
models.py — Pydantic request/response schemas matching the frontend contract in Esther.md.
"""

from pydantic import BaseModel, field_validator, Field
from typing import Optional, List, Literal
from datetime import datetime
import re

# Accepts both formats:
#   +25078XXXXXXX  (international, 13 chars)
#   078XXXXXXX     (local, 10 digits)
PHONE_REGEX = re.compile(r"^(\+250(72|73|78|79)\d{7}|0(72|73|78|79)\d{7})$")


def _validate_phone(value: str) -> str:
    cleaned = value.strip().replace(" ", "")
    if not PHONE_REGEX.match(cleaned):
        raise ValueError(
            "Inomero ya telefoni ntabwo ari yo / Phone number is invalid. "
            "Use +25078XXXXXXX or 078XXXXXXX (10 digits)"
        )
    return cleaned


def _validate_password(value: str) -> str:
    if len(value) < 6:
        raise ValueError(
            "Ijambo banga rigomba kuba rifite inyuguti 6 nibura / "
            "Password must be at least 6 characters."
        )
    return value


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class GuardianRegisterIn(BaseModel):
    name: str = Field(..., min_length=2, description="Full name of the guardian")
    phone: str = Field(..., description="Rwandan phone: +25078XXXXXXX or 078XXXXXXX")
    password: str = Field(..., min_length=6, description="Password, minimum 6 characters")
    relationship: str = Field(..., description="Relationship to blind user e.g. Mother, Father")

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v): return _validate_phone(v)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v): return _validate_password(v)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Izina rigomba kuba rifite inyuguti 2 nibura / Name must be at least 2 characters.")
        return v.strip()


class BlindRegisterIn(BaseModel):
    name: str = Field(..., min_length=2, description="Full name of the blind user")
    phone: str = Field(..., description="Rwandan phone: +25078XXXXXXX or 078XXXXXXX")
    emergency_phone: Optional[str] = Field(None, description="Emergency contact phone number")
    language: Literal["en", "rw"] = Field("en", description="Preferred language: en or rw")
    voice_speed: Literal["Slow", "Normal", "Fast"] = Field("Normal", description="Voice alert speed")

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v): return _validate_phone(v)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Izina rigomba kuba rifite inyuguti 2 nibura / Name must be at least 2 characters.")
        return v.strip()

    @field_validator("emergency_phone")
    @classmethod
    def validate_emergency_phone(cls, v):
        if v is not None:
            return _validate_phone(v)
        return v

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
    phone: str = Field(..., description="Rwandan phone: +25078XXXXXXX or 078XXXXXXX")
    password: str = Field(..., min_length=6, description="Account password")

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v): return _validate_phone(v)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v): return _validate_password(v)

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
