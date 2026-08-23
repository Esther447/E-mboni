"""
auth.py — JWT token creation and verification.
Preserved: bcrypt password hashing, JWT tokens, Bearer auth.
Changed: user lookup now uses Firestore via firestore_service.
"""

from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import HTTPException, Header
import os
import secrets

SECRET_KEY        = os.getenv("EMBONI_SECRET_KEY", secrets.token_hex(32))
ALGORITHM         = "HS256"
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


def get_current_user_from_token(token: str) -> dict:
    """Decode token and fetch user from Firestore. Returns user dict."""
    from firestore_service import get_user_by_id
    payload = decode_token(token)
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user
