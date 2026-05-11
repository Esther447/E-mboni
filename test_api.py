"""
test_api.py — E-mboni Integration Tests
Run with: python test_api.py
Server must be running on http://localhost:8000
"""

import requests
import json

BASE = "http://localhost:8000"

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

def show(r):
    status = r.status_code
    icon = "✅" if status < 400 else "❌"
    print(f"{icon} Status: {status}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


# ------------------------------------------------------------------
# TEST 1 — 422 validation: invalid phone + short password
# ------------------------------------------------------------------
section("TEST 1 — 422 Validation (invalid phone + short password)")
r = requests.post(f"{BASE}/auth/login", json={"phone": "0711234567", "password": "hi"})
show(r)

# ------------------------------------------------------------------
# TEST 2 — Blind user login
# ------------------------------------------------------------------
section("TEST 2 — Blind user login (blind123)")
r = requests.post(f"{BASE}/auth/login", json={"phone": "+250781000002", "password": "blind123"})
show(r)
blind_token = r.json().get("token", "")

# ------------------------------------------------------------------
# TEST 3 — Guardian login
# ------------------------------------------------------------------
section("TEST 3 — Guardian login (guardian123)")
r = requests.post(f"{BASE}/auth/login", json={"phone": "+250781000001", "password": "guardian123"})
show(r)
guardian_token = r.json().get("token", "")

# ------------------------------------------------------------------
# TEST 4 — Start session as blind user
# ------------------------------------------------------------------
section("TEST 4 — Start navigation session (blind user)")
r = requests.post(f"{BASE}/session/start", headers={"Authorization": f"Bearer {blind_token}"})
show(r)

# ------------------------------------------------------------------
# TEST 5 — Guardian dashboard (should show blind user + session)
# ------------------------------------------------------------------
section("TEST 5 — Guardian dashboard")
r = requests.get(f"{BASE}/guardian/dashboard", headers={"Authorization": f"Bearer {guardian_token}"})
show(r)

# ------------------------------------------------------------------
# TEST 6 — Admin login + list users
# ------------------------------------------------------------------
section("TEST 6 — Admin login + list users")
r = requests.post(f"{BASE}/auth/login", json={"phone": "+250780000000", "password": "admin123"})
show(r)
admin_token = r.json().get("token", "")

r = requests.get(f"{BASE}/users", headers={"Authorization": f"Bearer {admin_token}"})
data = r.json()
if isinstance(data, list):
    print(f"✅ Total users in DB: {len(data)}")
    for u in data:
        print(f"   [{u['role']}] {u['name']} — {u['phone']}")
else:
    print(f"❌ Unexpected response: {data}")

print("\n" + "="*55)
print("  All tests complete.")
print("="*55)
