"""
auth.py — JWT token creation and verification.
"""

from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import HTTPException, Header
from sqlalchemy.orm import Session
from database import User, get_db
import os
import secrets

# Load secret from environment variable.
# If not set, generate a random one for this session (safe for dev).
# In production: set SECRET_KEY in your environment before starting the server.
SECRET_KEY = os.getenv("EMBONI_SECRET_KEY", secrets.token_hex(32))
ALGORITHM  = "HS256"
TOKEN_EXPIRE_DAYS = 30


def create_token(user_id: int, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def get_current_user(authorization: str = Header(...), db: Session = next(get_db())):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user
