"""
init_db.py — Run this once to create all tables and seed demo accounts.

Usage:
    python init_db.py
"""

from database import init_db, SessionLocal, User

if __name__ == "__main__":
    print("Creating tables...")
    init_db()

    db = SessionLocal()
    users = db.query(User).all()
    db.close()

    print(f"Done. {len(users)} users seeded:")
    for u in users:
        print(f"  [{u.role.value}] {u.name} — {u.phone}")
