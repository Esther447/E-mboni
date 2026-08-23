"""
migrate_to_firestore.py — Migrate existing emboni.db (SQLite) data to Firestore.

Usage:
    python migrate_to_firestore.py

Safety:
- Checks for existing documents before inserting (idempotent — safe to run twice).
- Does NOT delete emboni.db.
- Reports counts at the end.
"""

import os
import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

import firebase_admin
from firebase_admin import credentials, firestore as fs_admin

# ---------------------------------------------------------------------------
# Init Firebase
# ---------------------------------------------------------------------------
cred_path = os.getenv("FIREBASE_CREDENTIALS")
if not cred_path:
    raise RuntimeError("FIREBASE_CREDENTIALS env variable not set.")

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = fs_admin.client()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dt(value) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def _ensure_counter(collection: str, value: int):
    """Set counter to max(current, value) so new inserts don't collide."""
    ref = db.collection("_counters").document(collection)
    doc = ref.get()
    current = doc.get("value") if doc.exists else 0
    if value > current:
        ref.set({"value": value})


# ---------------------------------------------------------------------------
# Migrate users
# ---------------------------------------------------------------------------

def migrate_users(conn) -> int:
    cursor = conn.execute("SELECT id, name, phone, password_hash, role, language, voice_speed, status, guardian_id, emergency_phone, relationship, created_at FROM users")
    rows = cursor.fetchall()
    migrated = 0
    max_id = 0

    for row in rows:
        uid = row[0]
        max_id = max(max_id, uid)
        ref = db.collection("users").document(str(uid))

        if ref.get().exists:
            print(f"  [SKIP] user id={uid} already exists")
            continue

        data = {
            "id":              uid,
            "name":            row[1],
            "phone":           row[2],
            "password_hash":   row[3],
            "role":            row[4],
            "language":        row[5] or "en",
            "voice_speed":     row[6] or "Normal",
            "status":          row[7] or "active",
            "guardian_id":     row[8],
            "emergency_phone": row[9],
            "relationship":    row[10],
            "created_at":      _to_dt(row[11]),
        }
        ref.set(data)
        print(f"  [OK] user id={uid} [{row[4]}] {row[1]} — {row[2]}")
        migrated += 1

    _ensure_counter("users", max_id)
    return migrated


# ---------------------------------------------------------------------------
# Migrate alerts
# ---------------------------------------------------------------------------

def migrate_alerts(conn) -> int:
    cursor = conn.execute("SELECT id, blind_id, message, level, created_at FROM alerts")
    rows = cursor.fetchall()
    migrated = 0
    max_id = 0

    for row in rows:
        aid = row[0]
        max_id = max(max_id, aid)
        ref = db.collection("alerts").document(str(aid))

        if ref.get().exists:
            print(f"  [SKIP] alert id={aid} already exists")
            continue

        data = {
            "id":         aid,
            "blind_id":   row[1],
            "message":    row[2],
            "level":      row[3],
            "is_read":    False,
            "created_at": _to_dt(row[4]),
        }
        ref.set(data)
        print(f"  [OK] alert id={aid} blind_id={row[1]} [{row[3]}]")
        migrated += 1

    _ensure_counter("alerts", max_id)
    return migrated


# ---------------------------------------------------------------------------
# Migrate sessions
# ---------------------------------------------------------------------------

def migrate_sessions(conn) -> int:
    cursor = conn.execute("SELECT id, blind_id, started_at, ended_at, status FROM sessions")
    rows = cursor.fetchall()
    migrated = 0
    max_id = 0

    for row in rows:
        sid = row[0]
        max_id = max(max_id, sid)
        ref = db.collection("sessions").document(str(sid))

        if ref.get().exists:
            print(f"  [SKIP] session id={sid} already exists")
            continue

        data = {
            "id":         sid,
            "blind_id":   row[1],
            "started_at": _to_dt(row[2]),
            "ended_at":   _to_dt(row[3]) if row[3] else None,
            "status":     row[4] or "ended",
        }
        ref.set(data)
        print(f"  [OK] session id={sid} blind_id={row[1]} [{row[4]}]")
        migrated += 1

    _ensure_counter("sessions", max_id)
    return migrated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "emboni.db")
    if not os.path.exists(db_path):
        print(f"emboni.db not found at {db_path} — nothing to migrate.")
        exit(0)

    print(f"\nConnecting to {db_path}...")
    conn = sqlite3.connect(db_path)

    print("\n--- Migrating users ---")
    u = migrate_users(conn)

    print("\n--- Migrating alerts ---")
    a = migrate_alerts(conn)

    print("\n--- Migrating sessions ---")
    s = migrate_sessions(conn)

    conn.close()

    print(f"\n{'='*40}")
    print(f"Migration complete.")
    print(f"  Users migrated:    {u}")
    print(f"  Alerts migrated:   {a}")
    print(f"  Sessions migrated: {s}")
    print(f"{'='*40}")
    print("\nemboni.db has NOT been deleted. Keep it as backup.")
