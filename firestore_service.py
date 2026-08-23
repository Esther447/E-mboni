"""
firestore_service.py — Firestore repository layer for E-mboni.
All database operations go through this module.
Preserves the same data contracts as the SQLAlchemy models.
"""

from datetime import datetime, timezone
from typing import Optional
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from firebase import get_firestore

# ---------------------------------------------------------------------------
# ID generation — Firestore uses string doc IDs.
# We use an atomic counter in a dedicated collection to preserve integer IDs
# for backward compatibility with the Flutter app.
# ---------------------------------------------------------------------------

def _next_id(collection: str) -> int:
    db = get_firestore()
    counter_ref = db.collection("_counters").document(collection)

    @firestore.transactional
    def _increment(transaction):
        snap = counter_ref.get(transaction=transaction)
        current = snap.get("value") if snap.exists else 0
        next_val = current + 1
        transaction.set(counter_ref, {"value": next_val})
        return next_val

    transaction = db.transaction()
    return _increment(transaction)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_dt(value) -> Optional[datetime]:
    """Convert Firestore DatetimeWithNanoseconds or None to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return value


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------

def _user_from_doc(doc) -> Optional[dict]:
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["id"] = data.get("id") or int(doc.id)
    data["created_at"] = _to_dt(data.get("created_at"))
    return data


# ---------------------------------------------------------------------------
# USERS
# ---------------------------------------------------------------------------

def create_user(
    name: str,
    phone: str,
    password_hash: str,
    role: str,
    language: str = "en",
    voice_speed: str = "Normal",
    status: str = "active",
    guardian_id: Optional[int] = None,
    emergency_phone: Optional[str] = None,
    relationship: Optional[str] = None,
) -> dict:
    db = get_firestore()
    user_id = _next_id("users")
    data = {
        "id":              user_id,
        "name":            name,
        "phone":           phone,
        "password_hash":   password_hash,
        "role":            role,
        "language":        language,
        "voice_speed":     voice_speed,
        "status":          status,
        "guardian_id":     guardian_id,
        "emergency_phone": emergency_phone,
        "relationship":    relationship,
        "created_at":      _now(),
    }
    db.collection("users").document(str(user_id)).set(data)
    return data


def get_user_by_id(user_id: int) -> Optional[dict]:
    db = get_firestore()
    doc = db.collection("users").document(str(user_id)).get()
    return _user_from_doc(doc)


def get_user_by_phone(phone: str) -> Optional[dict]:
    db = get_firestore()
    docs = db.collection("users").where(filter=FieldFilter("phone", "==", phone)).limit(1).stream()
    for doc in docs:
        return _user_from_doc(doc)
    return None


def get_all_users() -> list[dict]:
    db = get_firestore()
    return [_user_from_doc(d) for d in db.collection("users").stream()]


def get_blind_users_for_guardian(guardian_id: int) -> list[dict]:
    db = get_firestore()
    docs = db.collection("users").where(filter=FieldFilter("guardian_id", "==", guardian_id)).stream()
    return [_user_from_doc(d) for d in docs]


def get_users_by_role(role: str) -> list[dict]:
    db = get_firestore()
    docs = db.collection("users").where(filter=FieldFilter("role", "==", role)).stream()
    return [_user_from_doc(d) for d in docs]


def update_user_status(user_id: int, status: str) -> Optional[dict]:
    db = get_firestore()
    ref = db.collection("users").document(str(user_id))
    if not ref.get().exists:
        return None
    ref.update({"status": status})
    return _user_from_doc(ref.get())


# ---------------------------------------------------------------------------
# ALERTS
# ---------------------------------------------------------------------------

def _alert_from_doc(doc) -> Optional[dict]:
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["id"] = data.get("id") or int(doc.id)
    data["created_at"] = _to_dt(data.get("created_at"))
    return data


def create_alert(blind_id: int, message: str, level: str) -> dict:
    db = get_firestore()
    alert_id = _next_id("alerts")
    data = {
        "id":         alert_id,
        "blind_id":   blind_id,
        "message":    message,
        "level":      level,
        "is_read":    False,
        "created_at": _now(),
    }
    db.collection("alerts").document(str(alert_id)).set(data)
    return data


def get_alerts_for_blind(blind_id: int, limit: int = 50) -> list[dict]:
    db = get_firestore()
    docs = (
        db.collection("alerts")
        .where(filter=FieldFilter("blind_id", "==", blind_id))
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [_alert_from_doc(d) for d in docs]


def mark_alert_read(alert_id: int) -> Optional[dict]:
    """Mark a single alert as read. Returns updated alert or None if not found."""
    db = get_firestore()
    ref = db.collection("alerts").document(str(alert_id))
    doc = ref.get()
    if not doc.exists:
        return None
    ref.update({"is_read": True})
    return _alert_from_doc(ref.get())


def get_alert_by_id(alert_id: int) -> Optional[dict]:
    db = get_firestore()
    doc = db.collection("alerts").document(str(alert_id)).get()
    return _alert_from_doc(doc)


def get_alerts_for_blinds(blind_ids: list[int], limit: int = 10) -> list[dict]:
    """Get recent alerts for multiple blind users."""
    if not blind_ids:
        return []

    all_alerts = []

    for bid in blind_ids:
        all_alerts.extend(get_alerts_for_blind(bid, limit=limit))

    all_alerts.sort(
        key=lambda a: a["created_at"],
        reverse=True,
    )

    return all_alerts[:limit]


def get_all_alerts(limit: int = 100) -> list[dict]:
    db = get_firestore()
    docs = (
        db.collection("alerts")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [_alert_from_doc(d) for d in docs]


def get_last_alert_for_blind(blind_id: int) -> Optional[dict]:
    db = get_firestore()
    docs = (
        db.collection("alerts")
        .where(filter=FieldFilter("blind_id", "==", blind_id))
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for doc in docs:
        return _alert_from_doc(doc)
    return None


def get_alerts_since(since: datetime, limit: int = 10) -> list[dict]:
    db = get_firestore()
    docs = (
        db.collection("alerts")
        .where(filter=FieldFilter("created_at", ">=", since))
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [_alert_from_doc(d) for d in docs]


def count_alerts_since(since: datetime) -> int:
    db = get_firestore()
    docs = db.collection("alerts").where(filter=FieldFilter("created_at", ">=", since)).stream()
    return sum(1 for _ in docs)


def get_alerts_with_user(limit: int = 200) -> list[dict]:
    """Returns alerts joined with user name — used by admin logs."""
    alerts = get_all_alerts(limit=limit)
    result = []
    for a in alerts:
        user = get_user_by_id(a["blind_id"])
        result.append({**a, "user_name": user["name"] if user else "Unknown"})
    return result


# ---------------------------------------------------------------------------
# SESSIONS
# ---------------------------------------------------------------------------

def _session_from_doc(doc) -> Optional[dict]:
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["id"] = data.get("id") or int(doc.id)
    data["started_at"] = _to_dt(data.get("started_at"))
    data["ended_at"]   = _to_dt(data.get("ended_at"))
    return data


def start_session(blind_id: int) -> dict:
    db = get_firestore()
    # End any existing active session first
    _end_active_sessions(blind_id)
    session_id = _next_id("sessions")
    data = {
        "id":         session_id,
        "blind_id":   blind_id,
        "started_at": _now(),
        "ended_at":   None,
        "status":     "active",
    }
    db.collection("sessions").document(str(session_id)).set(data)
    return data


def end_session(blind_id: int) -> Optional[dict]:
    db = get_firestore()
    docs = (
        db.collection("sessions")
        .where(filter=FieldFilter("blind_id", "==", blind_id))
        .where(filter=FieldFilter("status", "==", "active"))
        .limit(1)
        .stream()
    )
    for doc in docs:
        ref = db.collection("sessions").document(doc.id)
        ref.update({"status": "ended", "ended_at": _now()})
        return _session_from_doc(ref.get())
    return None


def _end_active_sessions(blind_id: int):
    """End all active sessions for a blind user (called before starting new one)."""
    db = get_firestore()
    docs = (
        db.collection("sessions")
        .where(filter=FieldFilter("blind_id", "==", blind_id))
        .where(filter=FieldFilter("status", "==", "active"))
        .stream()
    )
    for doc in docs:
        db.collection("sessions").document(doc.id).update({
            "status": "ended",
            "ended_at": _now(),
        })


def get_active_session(blind_id: int) -> Optional[dict]:
    db = get_firestore()
    docs = (
        db.collection("sessions")
        .where(filter=FieldFilter("blind_id", "==", blind_id))
        .where(filter=FieldFilter("status", "==", "active"))
        .limit(1)
        .stream()
    )
    for doc in docs:
        return _session_from_doc(doc)
    return None


def get_active_session_ids() -> list[int]:
    """Returns blind_ids of all currently active sessions — used by admin overview."""
    db = get_firestore()
    docs = db.collection("sessions").where(filter=FieldFilter("status", "==", "active")).stream()
    return [d.to_dict().get("blind_id") for d in docs]


def get_all_sessions(limit: int = 100) -> list[dict]:
    db = get_firestore()
    docs = (
        db.collection("sessions")
        .order_by("started_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [_session_from_doc(d) for d in docs]
