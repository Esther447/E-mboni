"""
init_db.py — Seed demo accounts into Firestore.
Run once: python init_db.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from database import hash_password
import firestore_service as fs

DEMO_USERS = [
    {
        "name": "Admin User",
        "phone": "+250780000000",
        "password": "admin123",
        "role": "admin",
    },
    {
        "name": "Guardian Demo",
        "phone": "+250781000001",
        "password": "guardian123",
        "role": "guardian",
        "relationship": "Parent",
    },
    {
        "name": "Blind Demo",
        "phone": "+250781000002",
        "password": "blind123",
        "role": "blind",
        "language": "en",
        "voice_speed": "Normal",
    },
]

if __name__ == "__main__":
    print("Seeding demo accounts into Firestore...")

    guardian_id = None
    for u in DEMO_USERS:
        existing = fs.get_user_by_phone(u["phone"])
        if existing:
            print(f"  [SKIP] {u['phone']} already exists (id={existing['id']})")
            if u["role"] == "guardian":
                guardian_id = existing["id"]
            continue

        kwargs = {
            "name":          u["name"],
            "phone":         u["phone"],
            "password_hash": hash_password(u["password"]),
            "role":          u["role"],
        }
        if u["role"] == "guardian":
            kwargs["relationship"] = u.get("relationship", "")
        if u["role"] == "blind":
            kwargs["language"]       = u.get("language", "en")
            kwargs["voice_speed"]    = u.get("voice_speed", "Normal")
            kwargs["guardian_id"]    = guardian_id
            kwargs["emergency_phone"] = "+250781000001"

        created = fs.create_user(**kwargs)
        if u["role"] == "guardian":
            guardian_id = created["id"]
        print(f"  [OK] [{created['role']}] {created['name']} — {created['phone']} (id={created['id']})")

    print("Done.")
