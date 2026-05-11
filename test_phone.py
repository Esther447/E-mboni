import re

PHONE_REGEX = re.compile(r"^(\+250(72|73|78|79)\d{7}|0(72|73|78|79)\d{7})$")

tests = [
    ("+250781234567", True,  "international MTN"),
    ("+250791234567", True,  "international Airtel"),
    ("+250721234567", True,  "international 72"),
    ("+250731234567", True,  "international 73"),
    ("0781234567",    True,  "local 10 digits MTN"),
    ("0791234567",    True,  "local 10 digits Airtel"),
    ("0721234567",    True,  "local 10 digits 72"),
    ("0731234567",    True,  "local 10 digits 73"),
    ("0788000000",    True,  "local 10 digits valid"),
    ("078800000",     False, "too short — 9 digits"),
    ("07880000000",   False, "too long — 11 digits"),
    ("0711000002",    False, "invalid prefix 71"),
    ("0751234567",    False, "invalid prefix 75"),
    ("788000000",     False, "missing leading 0"),
    ("+250711000002", False, "international invalid prefix"),
]

all_pass = True
for phone, expected, label in tests:
    result = bool(PHONE_REGEX.match(phone.replace(" ", "")))
    ok = result == expected
    if not ok:
        all_pass = False
    icon = "✅" if ok else "❌"
    print(f"  {icon} {phone:22} expected={str(expected):5} got={str(result):5}  {label}")

print()
print("✅ All tests passed!" if all_pass else "❌ SOME TESTS FAILED")

# Also test the actual validator from models.py
print("\n--- Testing models.py validator ---")
from models import LoginRequest
try:
    LoginRequest(phone="0788000000", password="hi")
    print("❌ Should have raised for short password")
except Exception as e:
    print(f"✅ Caught validation error: {e}")

try:
    LoginRequest(phone="0711000002", password="blind123")
    print("❌ Should have raised for invalid prefix")
except Exception as e:
    print(f"✅ Caught validation error: {e}")

try:
    r = LoginRequest(phone="0781234567", password="blind123")
    print(f"✅ Valid local format accepted: {r.phone}")
except Exception as e:
    print(f"❌ Should have passed: {e}")

try:
    r = LoginRequest(phone="+250781234567", password="blind123")
    print(f"✅ Valid international format accepted: {r.phone}")
except Exception as e:
    print(f"❌ Should have passed: {e}")
