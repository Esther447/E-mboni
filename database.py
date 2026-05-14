"""
database.py — PostgreSQL database setup with SQLAlchemy ORM.
Tables: users, alerts, sessions
Seeded with 3 demo accounts from Esther.md spec.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import declarative_base, sessionmaker, relationship as orm_relationship
from datetime import datetime
import enum
import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./emboni.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Enums
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
# Models
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String, nullable=False)
    phone           = Column(String, unique=True, nullable=False, index=True)
    password_hash   = Column(String, nullable=False)
    role            = Column(SAEnum(RoleEnum), nullable=False)
    language        = Column(SAEnum(LanguageEnum), default=LanguageEnum.en)
    voice_speed     = Column(SAEnum(VoiceSpeedEnum), default=VoiceSpeedEnum.Normal)
    status          = Column(SAEnum(StatusEnum), default=StatusEnum.active)
    guardian_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    emergency_phone = Column(String, nullable=True)
    relationship    = Column(String, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    blind_users     = orm_relationship("User", foreign_keys="[User.guardian_id]")


class Alert(Base):
    __tablename__ = "alerts"

    id         = Column(Integer, primary_key=True, index=True)
    blind_id   = Column(Integer, ForeignKey("users.id"), nullable=False)
    message    = Column(String, nullable=False)
    level      = Column(SAEnum(AlertLevelEnum), nullable=False, default=AlertLevelEnum.warning)
    created_at = Column(DateTime, default=datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id         = Column(Integer, primary_key=True, index=True)
    blind_id   = Column(Integer, ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at   = Column(DateTime, nullable=True)
    status     = Column(SAEnum(SessionStatusEnum), default=SessionStatusEnum.active)


# ---------------------------------------------------------------------------
# DB Dependency
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Init + Seed
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=10)).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def init_db():
    Base.metadata.create_all(bind=engine)
