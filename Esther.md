# E-mboni — Complete Developer Guide
> Read this entire file before touching anything in this project.
> Written for a new developer who has never seen this codebase.

---

## What Is This Project?

E-mboni is an AI-powered navigation assistant for visually impaired people in Rwanda.

A blind user holds their phone in front of them while walking. The app captures camera frames, sends them to this backend, and the backend detects obstacles (cars, stairs, people, chairs, etc.) using two YOLO models + a MiDaS depth model. The backend responds with structured JSON that tells the app what to speak aloud and whether to vibrate.

There is also a **guardian** (caregiver/parent) who gets a live alert dashboard to monitor their blind user from their own phone.

**What the blind user hears:** "moving car, 3.5 meters ahead" or "stairs below, watch your step."

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12 + FastAPI |
| Database | SQLite (local dev) or PostgreSQL (production) |
| ORM | SQLAlchemy |
| Auth | JWT tokens via python-jose (30-day expiry) |
| AI — object detection | YOLOv8 (two models run on every frame) |
| AI — depth estimation | MiDaS (blended with bbox for distance accuracy) |
| Frontend | React Native + Expo (separate repo, NOT in this folder) |

---

## Project File Structure

```
E-mboni/
│
├── main.py              ← THE server. Every API endpoint is here.
├── database.py          ← DB connection + all ORM table models
├── auth.py              ← JWT create/decode
├── models.py            ← Pydantic request/response schemas + phone validation
├── spatial_engine.py    ← YOLO bbox → direction + distance + speech text
├── memory.py            ← Consistency filter, crowd detector, motion/TTC engine
├── events.py            ← Privacy-safe in-memory safety event store
├── roles.py             ← Role permission definitions
│
├── queries/
│   └── create_tables.sql  ← Raw SQL schema for PostgreSQL (run once)
│
├── init_db.py           ← Run once to create tables + seed 3 demo accounts
├── test_api.py          ← Integration tests for all endpoints
├── test_phone.py        ← Phone number format validation tests
├── requirements.txt     ← All Python dependencies
├── Dockerfile           ← Docker config for deployment
├── .env                 ← Local config (DATABASE_URL + SECRET_KEY) — not in git
│
├── yolov8n.onnx         ← Custom trained model: 7 indoor classes
├── yolov8m.pt           ← General YOLOv8 medium model: 80 COCO classes
├── yolo_detection.py    ← Local webcam detection script (dev/testing only)
├── train_model.py       ← Fine-tunes YOLOv8 on custom dataset
├── export_model.py      ← Copies best.onnx to root after training
│
├── train/               ← Training images + labels
├── valid/               ← Validation images + labels
├── test/                ← Test images + labels
└── runs/                ← Training output (weights, plots, metrics)
```

---

## Three User Roles

| Role | Who | What they can do |
|------|-----|-----------------|
| `blind` | The visually impaired user | Use camera detection, start/end sessions |
| `guardian` | Caregiver / parent | See alert dashboard, track sessions |
| `admin` | System admin | Manage all users, view all alerts/sessions |

**Privacy rules (enforced in code):**
- Guardian receives TEXT-ONLY alerts. No camera feed, no real-time location.
- Admin sees user accounts and device status only. No personal data, no camera.
- Blind user controls their own location sharing.

---

## Database Tables

### `users`
| Column | Type | Notes |
|--------|------|-------|
| id | Integer | Primary key |
| name | String | Full name |
| phone | String | Unique. Rwandan format only: `+25078XXXXXXX` |
| password_hash | String | bcrypt hashed |
| role | Enum | `blind` / `guardian` / `admin` |
| language | Enum | `en` / `rw` |
| voice_speed | Enum | `Slow` / `Normal` / `Fast` |
| status | Enum | `active` / `inactive` |
| guardian_id | FK → users | Only set for blind users, points to their guardian |
| emergency_phone | String | Optional, blind users only |
| relationship | String | Guardian's relation to blind user e.g. "Mother" |
| created_at | DateTime | Auto set |

### `alerts`
| Column | Type | Notes |
|--------|------|-------|
| id | Integer | Primary key |
| blind_id | FK → users | The blind user who triggered the alert |
| message | String | e.g. "STOP. car straight ahead" |
| level | Enum | `safe` / `warning` / `danger` |
| created_at | DateTime | Auto set |

### `sessions`
| Column | Type | Notes |
|--------|------|-------|
| id | Integer | Primary key |
| blind_id | FK → users | The blind user |
| started_at | DateTime | When navigation started |
| ended_at | DateTime | Null = session still active |
| status | Enum | `active` / `ended` |

---

## How to Set Up and Run (Local Dev)

### Step 1 — Clone and enter the project
```bash
cd E-mboni
```

### Step 2 — Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows
```

### Step 3 — Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Create the .env file
Create a file called `.env` in the project root with this content:
```
DATABASE_URL=sqlite:///./emboni.db
SECRET_KEY=your-secret-key-change-in-production
```
To use PostgreSQL instead of SQLite:
```
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/emboni
SECRET_KEY=your-secret-key-change-in-production
```

### Step 5 — Create tables and seed demo accounts
```bash
venv/bin/python init_db.py
```

### Step 6 — Start the server
```bash
venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server: `http://localhost:8000`
Swagger UI (interactive docs): `http://localhost:8000/docs`

> Note: first startup takes ~60 seconds because MiDaS downloads its weights (~50MB) on first run. Subsequent starts are instant (cached).

---

## Demo Accounts

| Role | Phone | Password | Navigates to |
|------|-------|----------|-------------|
| admin | +250780000000 | admin123 | Admin dashboard |
| guardian | +250781000001 | guardian123 | Guardian dashboard |
| blind | +250781000002 | blind123 | Navigation screen |

---

## How the Detection Pipeline Works

This is the most important part to understand. When `POST /detect` receives a camera frame:

```
JPEG frame received
        │
        ▼
Decode with OpenCV
        │
        ▼
Run yolov8n.onnx (custom 7-class model, conf=0.6)
Run yolov8m.pt   (general 80-class model, conf=0.6)
        │
        ▼
Merge + deduplicate results by label
        │
        ▼
ConsistencyFilter — object must appear 2+ frames in a row
(prevents ghost/false detections from being announced)
        │
        ▼
spatial_engine.process_detections()
— assigns direction (left/center/right) from bbox center x position
— assigns distance using bbox area heuristic
        │
        ▼
MiDaS depth model — gets depth score at object center pixel
Blended with bbox heuristic: final_dist = bbox*0.5 + midas*0.5
        │
        ▼
CrowdDetector — if 3+ people within ~2m, collapse to crowd summary
        │
        ▼
Assign dangerLevel per object (see danger rules below)
Sort: danger → warning → safe
        │
        ▼
If blind user token present + top object is danger/warning:
  → save alert to DB (guardian sees it)
        │
        ▼
Build summary string for voice
Return JSON response to frontend
```

### Danger Level Rules

```
Any object < 0.3m          → always "danger"
Any object < 0.6m          → "danger" if HIGH category, else "warning"

HIGH objects (car, truck, bus, motorcycle, vehicle, van):
  distance <= 5m  → "danger"
  else            → "warning"

MEDIUM objects (person, bicycle, chair, stairs, dog, cat, bench, pole...):
  distance <= 1.5m → "danger"
  distance <= 4m   → "warning"
  else             → "safe"

Critical objects (car, stair, truck, bus, motorcycle, stairs):
  distance <= 1.5m → always "danger" regardless of category
```

### Distance Calculation

MiDaS depth score is blended with a bounding box area heuristic:

| Normalized bbox area | Estimated distance |
|---------------------|--------------------|
| > 0.40 | 0.5m |
| > 0.20 | 1.0m |
| > 0.10 | 2.0m |
| > 0.05 | 3.5m |
| > 0.02 | 6.0m |
| > 0.01 | 10.0m |
| else | 15.0m |

Final distance = `bbox_estimate * 0.5 + midas_estimate * 0.5`

### Direction Mapping

| Object center x (normalized) | Direction |
|------------------------------|-----------|
| 0.0 – 0.2 | far left → `"left"` |
| 0.2 – 0.4 | left → `"left"` |
| 0.4 – 0.6 | center → `"center"` |
| 0.6 – 0.8 | right → `"right"` |
| 0.8 – 1.0 | far right → `"right"` |

### AI Models

| Model | File | Classes | Purpose |
|-------|------|---------|---------|
| Custom | `yolov8n.onnx` | 7 | Indoor: bed, door, sofa, stair, table, toilet, person |
| General | `yolov8m.pt` | 80 | COCO: cars, people, animals, furniture, etc. |
| Depth | MiDaS small | — | Monocular depth estimation for distance accuracy |

---

## Intelligence Engine (memory.py)

| Feature | How it works |
|---------|-------------|
| ConsistencyFilter | Object must appear 2+ consecutive frames before being reported. Prevents one-frame ghost detections from triggering voice alerts. |
| CrowdDetector | If 3+ people detected within ~2m simultaneously, individual "person" alerts are collapsed into one crowd message: "Crowd ahead, navigate right. 3 people within 2 meters." HIGH hazards (car, stair) always bypass crowd mode. |
| MemoryEngine | Tracks object history across frames. Suppresses stationary objects for 20 seconds after first report. |
| TTC (Time-to-Collision) | If bbox area grows 20%+ in 2 frames → imminent collision, immediate STOP alert regardless of other rules. |
| Vibration lock | Stair/ground hazard in bottom 20% of frame → continuous vibration until clear. |
| Path clear | Requires 10 consecutive clean frames + 5 seconds since last DB alert before announcing "Path clear." |

---

## All API Endpoints

Base URL: `http://localhost:8000`
Protected endpoints require: `Authorization: Bearer <token>`

### POST /auth/register
Creates guardian + blind user pair together. No token required.

Request:
```json
{
  "guardian": {
    "name": "Sarah Kamau",
    "phone": "+250781000001",
    "password": "guardian123",
    "relationship": "Mother"
  },
  "blind_user": {
    "name": "James Kamau",
    "phone": "+250781000002",
    "language": "en",
    "voice_speed": "Normal"
  }
}
```
Response 201:
```json
{
  "guardian": { "id": 1, "name": "Sarah Kamau", "phone": "+250781000001", "role": "guardian" },
  "blind_user": { "id": 2, "name": "James Kamau", "phone": "+250781000002", "role": "blind", "guardian_id": 1 },
  "token": "<jwt>"
}
```

---

### POST /auth/login
No token required.

Request: `{ "phone": "+250781000001", "password": "guardian123" }`

Response 200 (guardian):
```json
{
  "token": "<jwt>",
  "user": {
    "id": 1, "name": "Sarah Kamau", "role": "guardian",
    "language": "en", "voice_speed": "Normal",
    "blind_user": { "id": 2, "name": "James Kamau", "status": "active", "language": "en" }
  }
}
```
For blind login: same shape, `blind_user` is `null`, `role` is `"blind"`.

Error 401: `{ "detail": "Wrong phone or password." }`

---

### GET /auth/me
Returns current user profile. Requires any valid token.

---

### POST /detect
The core endpoint. Send a camera frame, get detected objects back.

Request: `multipart/form-data`, field name `file`, JPEG image.
Optional: `Authorization: Bearer <token>` — if provided and the user is blind, danger/warning alerts are auto-saved to DB.

Response 200:
```json
{
  "objects": [
    {
      "name": "car",
      "isMoving": true,
      "distanceMeters": 3.5,
      "direction": "left",
      "dangerLevel": "danger"
    }
  ],
  "summary": "moving car, 3.5 meters ahead",
  "topDanger": "danger"
}
```

`direction`: `"left"` | `"center"` | `"right"`
`dangerLevel`: `"safe"` | `"warning"` | `"danger"`
`summary`: ready-to-speak string for the voice engine
`topDanger`: the worst danger level across all detected objects

---

### GET /guardian/dashboard
Requires guardian token.

Returns guardian profile, linked blind user info, last 10 alerts, active session.

```json
{
  "guardian": { "id": 1, "name": "Sarah Kamau", ... },
  "blind_user": { "id": 2, "name": "James Kamau", "status": "active", ... },
  "recent_alerts": [
    { "id": 1, "message": "STOP. car straight ahead", "level": "danger", "created_at": "..." }
  ],
  "active_session": { "id": 1, "started_at": "...", "ended_at": null, "status": "active" }
}
```

---

### GET /guardian/tracking
Requires guardian token.

Returns session timeline for the tracking screen.

```json
{
  "blind_name": "James Kamau",
  "blind_phone": "+250781000002",
  "is_scanning": true,
  "timeline": [
    { "time": "10:37", "event": "STOP. car straight ahead", "level": "danger" }
  ],
  "session": { "status": "Danger", "duration_minutes": 8, "alert_count": 3 }
}
```

---

### GET /alerts
Requires guardian or admin token.
- Guardian → last 50 alerts for their linked blind user
- Admin → last 100 alerts across all users

---

### GET /alerts/{blind_id}
Requires admin token. All alerts for one specific blind user.

---

### POST /session/start
Requires blind token. Starts a new navigation session. Auto-ends any previously active session.

---

### POST /session/end
Requires blind token. Ends the current active session.

---

### GET /session/active
Requires blind or guardian token. Returns the active session or null.

---

### GET /session/history
Requires admin token. Returns last 100 sessions.

---

### GET /users
Requires admin token. Returns all users.

---

### GET /users/{user_id}
Requires admin token. Returns one user by ID.

---

### PATCH /users/{user_id}/status?status=inactive
Requires admin token. Toggle user status: `active` or `inactive`.

---

### GET /location/last-known?user_id=X
Returns last known location if user has enabled sharing. Guardian-facing.

### POST /location/share?user_id=X&enabled=true
Blind user controls whether their location is shared with guardian.

### GET /devices/status?device_id=X
Returns `"online"` or `"offline"` based on last ping. Admin-facing.

### POST /devices/ping?device_id=X
Heartbeat from device. Marks it as online.

---

## Phone Number Validation

Only Rwandan phone numbers are accepted. Validated in `models.py`.

Accepted:
```
+250780000000   international MTN
+250790000000   international Airtel
+250720000000   international MTN
+250730000000   international Airtel
0780000000      local 10 digits (any of above prefixes)
```

Rejected:
```
0711234567      invalid prefix (71 not supported)
078123456       too short (9 digits)
+1234567890     not Rwandan
```

Error message is bilingual (English + Kinyarwanda):
```
Inomero ya telefoni ntabwo ari yo / Phone number is invalid.
Use +25078XXXXXXX or 078XXXXXXX (10 digits)
```

---

## Frontend Integration — What Needs to Be Done

The backend is fully ready. The React Native frontend still has hardcoded mock data. Here is every screen that needs to be wired up:

| Screen | File | Replace | With |
|--------|------|---------|------|
| Login | `app/(auth)/login.tsx` | `DEMO_ACCOUNTS` + `handleLogin()` | `POST /auth/login` |
| Guardian register | `app/(auth)/guardian-register.tsx` | `handleNext()` | `POST /auth/register` |
| Blind register | `app/(auth)/blind-register.tsx` | `handleDone()` | `POST /auth/register` |
| Guardian home | `app/(guardian)/index.tsx` | `USER` + `ALERTS` | `GET /guardian/dashboard` |
| Alerts screen | `app/(guardian)/alerts.tsx` | `ALERTS` | `GET /alerts` |
| Tracking screen | `app/(guardian)/tracking.tsx` | `ACTIVITY` | `GET /guardian/tracking` |
| Admin users | `app/(admin)/users.tsx` | `USERS` | `GET /users` |
| Navigation screen | `app/(tabs)/navigation.tsx` | local loop | `POST /detect` + session start/end |

### Token storage
```typescript
// After login, save token:
await AsyncStorage.setItem("token", data.token);

// On every protected request:
const token = await AsyncStorage.getItem("token");
headers: { "Authorization": `Bearer ${token}` }
```

### Base URL
```typescript
// services/api.ts
export const API_BASE = "http://<your-computer-lan-ip>:8000";
// Example: "http://192.168.1.5:8000"
// Do NOT use "localhost" from the phone — use the computer's WiFi IP
// Find it with: ip addr show (Linux) or ipconfig (Windows)
```

### Sending a camera frame to /detect
```typescript
const formData = new FormData();
formData.append("file", {
  uri: frameUri,
  type: "image/jpeg",
  name: "frame.jpg",
} as any);

const response = await fetch(`${API_BASE}/detect`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${token}` },
  body: formData,
});
const result = await response.json();
// result.summary  → speak this
// result.topDanger → "danger" / "warning" / "safe" → vibrate accordingly
```

### Starting and ending a session
```typescript
// When navigation screen opens (blind user only):
await fetch(`${API_BASE}/session/start`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${token}` },
});

// When navigation screen closes:
await fetch(`${API_BASE}/session/end`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${token}` },
});
```

### Polling guardian dashboard (every 5 seconds)
```typescript
useEffect(() => {
  const interval = setInterval(async () => {
    const token = await AsyncStorage.getItem("token");
    const res = await fetch(`${API_BASE}/guardian/dashboard`, {
      headers: { "Authorization": `Bearer ${token}` },
    });
    const data = await res.json();
    setAlerts(data.recent_alerts);
    setBlindUser(data.blind_user);
    setSession(data.active_session);
  }, 5000);
  return () => clearInterval(interval);
}, []);
```

### Error handling
All errors come back as:
```json
{ "detail": "Wrong phone or password." }
```
Validation errors (422):
```json
{ "detail": [{ "loc": ["body", "phone"], "msg": "Phone number is invalid..." }] }
```

---

## Missing Endpoints (Not Yet Built)

These screens exist in the frontend but the backend endpoints are not yet implemented:

| Endpoint | Screen |
|----------|--------|
| `GET /guardian/settings` | Guardian settings screen |
| `PATCH /guardian/settings` | Guardian settings update |
| `GET /admin/overview` | Admin overview with stats |

---

## Dependencies (requirements.txt)

```
fastapi
uvicorn
python-multipart
opencv-python-headless
onnxruntime
sqlalchemy
bcrypt
python-jose[cryptography]
psycopg2-binary
python-dotenv
torch --index-url https://download.pytorch.org/whl/cpu
timm
ultralytics
```

> `timm` is required by MiDaS. `torch` is pinned to CPU build to avoid downloading 900MB+ CUDA packages.

---

## Known Issues / Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: timm` | MiDaS dependency missing | `pip install timm` |
| `Address already in use` on port 8000 | Old server still running | `fuser -k 8000/tcp` |
| First startup very slow | MiDaS downloading weights (~50MB) | Wait ~60s, it only downloads once |
| `Connection refused` on phone | Using `localhost` instead of LAN IP | Use computer's IP: `192.168.x.x:8000` |
| `401 Unauthorized` | Token missing or expired | Login again and save the new token |
| `403 Forbidden` | Wrong role for that endpoint | Use the correct demo account |
| `422 Validation Error` | Wrong phone format or short password | Phone must be Rwandan format, password 6+ chars |

---

## Current Status

| Component | Status |
|-----------|--------|
| All backend endpoints | Done |
| Database (SQLite local / PostgreSQL prod) | Done |
| JWT authentication | Done |
| YOLOv8 detection (2 models) | Done |
| MiDaS depth blending | Done |
| Crowd detection | Done |
| Consistency filter | Done |
| Privacy model | Done |
| Demo accounts seeded | Done |
| Virtual environment | Done |
| Frontend wiring | Not started |
| `GET /guardian/settings` | Not built |
| `PATCH /guardian/settings` | Not built |
| `GET /admin/overview` | Not built |
| Production deployment | Not done |
