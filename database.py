"""
database.py — E-mboni database layer.
Enums, password hashing, and a compatibility shim so existing imports don't break.
SQLAlchemy/PostgreSQL replaced by Firebase Cloud Firestore (see firestore_service.py).
"""

import enum
import bcrypt


# ---------------------------------------------------------------------------
# Enums — unchanged, used throughout main.py and models.py
# ---------------------------------------------------------------------------

class RoleEnum(str, enum.Enum):
    blind    = "blind"
    guardian = "guardian"
    admin    = "admin"

class LanguageEnum(str, enum.Enum):
    en = "en"
    rw = "rw"

class VoiceSpeedEnum(str, enum.Enum):
    Slow   = "Slow"
    Normal = "Normal"
    Fast   = "Fast"

class StatusEnum(str, enum.Enum):
    active   = "active"
    inactive = "inactive"

class AlertLevelEnum(str, enum.Enum):
    safe    = "safe"
    warning = "warning"
    danger  = "danger"

class SessionStatusEnum(str, enum.Enum):
    active = "active"
    ended  = "ended"


# ---------------------------------------------------------------------------
# Password hashing — bcrypt preserved
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=10)).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# Compatibility shims — main.py imports these names from database.py.
# They now delegate to firestore_service.py.
# ---------------------------------------------------------------------------

def init_db():
    """No-op: Firestore collections are created on first write."""
    pass

def get_db():
    """
    Compatibility shim — main.py uses Depends(get_db) in many endpoints.
    Returns None since Firestore operations are handled directly in firestore_service.
    This generator is kept so existing Depends(get_db) signatures don't break.
    """
    yield None
