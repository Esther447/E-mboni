# E-mboni — Project Documentation
> Last updated: Phase 2 Complete — Backend fully operational with PostgreSQL.
> Read this before touching any file.

---

## Current Status — What Is Done Right Now

| Area | Status | Detail |
|---|---|---|
| YOLOv8 object detection | ✅ Done | Two models running, merged results |
| `/detect` frontend format | ✅ Done | Returns `objects`, `summary`, `topDanger` |
| PostgreSQL connected | ✅ Done | `e-mboni` database on localhost:5432 |
| `users` table | ✅ Done | 3 demo accounts seeded |
| `alerts` table | ✅ Done | Auto-saved on danger/warning detection |
| `sessions` table | ✅ Done | Start/end tracked with timestamps |
| `POST /auth/register` | ✅ Done | Creates guardian + blind user together |
| `POST /auth/login` | ✅ Done | Returns JWT token + nested user data |
| JWT authentication | ✅ Done | 30-day tokens, role-based access |
| `/guardian/dashboard` | ✅ Done | Privacy-safe subquery, last 10 alerts |
| `/session/start` `/session/end` | ✅ Done | Blind user navigation sessions |
| `/users/*` admin endpoints | ✅ Done | List, get, activate/deactivate users |
| Phone number validation | ✅ Done | Rwandan format only (072/073/078/079) |
| Bilingual error messages | ✅ Done | English + Kinyarwanda on 422 errors |
| Anti-chatter cooldown | ✅ Done | Per-object+position cooldown timer |
| Inference speed | ✅ Done | Reduced imgsz 640→320, ~2x faster |
| Danger alert → DB | ✅ Done | STRONG vibration triggers DB write |
| Crowd detection | ✅ Done | 3+ people → crowd summary message |
| Path clear DB check | ✅ Done | 5s silence rule before announcing clear |
| Integration tests | ✅ Done | `test_api.py` covers all 6 endpoints |
| Phone regex tests | ✅ Done | `test_phone.py` covers 15 formats |

### 🔜 Next — Frontend Integration (Phase 3)
The backend is fully ready. The frontend team can now connect all screens.

---

## Demo Accounts (Live in PostgreSQL)

| Role | Phone | Password | Screen after login |
|---|---|---|---|
| admin | +250780000000 | admin123 | Admin dashboard |
| guardian | +250781000001 | guardian123 | Guardian dashboard |
| blind | +250781000002 | blind123 | Navigation screen |

---

## Project File Structure

```
E-mboni/
│
├── main.py               ← FastAPI server — ALL endpoints live here
├── database.py           ← PostgreSQL connection, ORM models, seed accounts
├── auth.py               ← JWT token creation and verification
├── models.py             ← Pydantic schemas + phone/password validators
├── spatial_engine.py     ← YOLO bbox → direction, distance, speech
├── events.py             ← In-memory safety event store
├── memory.py             ← Anti-chatter, crowd detection, TTC engine
├── roles.py              ← Role permission definitions
│
├── queries/
│   └── create_tables.sql ← PostgreSQL schema — run once to create all tables
│
├── init_db.py            ← Run once to create tables + seed demo accounts
├── test_api.py           ← Integration tests for all 6 endpoints
├── test_phone.py         ← Phone regex validation tests (15 cases)
├── requirements.txt      ← All Python dependencies
├── Dockerfile            ← Docker config for deployment
│
├── yolo_detection.py     ← Local webcam detection engine (dev/testing)
├── yolov8n.onnx          ← Custom trained model (indoor classes)
├── yolov8n.pt            ← General pretrained model (80 COCO classes)
│
├── train_model.py        ← Fine-tunes YOLOv8 on custom dataset
├── export_model.py       ← Copies best.onnx to root after training
│
├── train/                ← Training images + labels
├── valid/                ← Validation images + labels
├── test/                 ← Test images + labels
└── runs/                 ← Training output (weights, plots, metrics)
```

---

## How to Run

### Start the server
```bash
python main.py
```
Server: `http://localhost:8000`
Swagger docs: `http://localhost:8000/docs`

### Run integration tests (server must be running)
```bash
python test_api.py
```

### Run phone validation tests
```bash
python test_phone.py
```

### Run local webcam detection
```bash
python yolo_detection.py
```

---

## Database — PostgreSQL

**Connection string:**
```
postgresql://postgres:<password>@localhost:5432/e-mboni
```

### Tables

#### `users`
| Column | Type | Notes |
|---|---|---|
| id | SERIAL | Primary key |
| name | VARCHAR | Full name |
| phone | VARCHAR | Unique. Format: `+25078XXXXXXX` or `078XXXXXXX` |
| password_hash | TEXT | bcrypt hashed |
| role | ENUM | `blind` / `guardian` / `admin` |
| language | ENUM | `en` / `rw` |
| voice_speed | ENUM | `Slow` / `Normal` / `Fast` |
| status | ENUM | `active` / `inactive` |
| guardian_id | FK → users | Only for blind users |
| emergency_phone | VARCHAR | Optional, blind users only |
| relationship | VARCHAR | Guardian's relationship to blind user |
| created_at | TIMESTAMP | Auto |

#### `alerts`
| Column | Type | Notes |
|---|---|---|
| id | SERIAL | Primary key |
| blind_id | FK → users | The blind user who triggered the alert |
| message | TEXT | e.g. `"STOP. car on your left"` |
| level | ENUM | `safe` / `warning` / `danger` |
| created_at | TIMESTAMP | Auto |

#### `sessions`
| Column | Type | Notes |
|---|---|---|
| id | SERIAL | Primary key |
| blind_id | FK → users | The blind user |
| started_at | TIMESTAMP | When navigation started |
| ended_at | TIMESTAMP | Null = session still active |
| status | ENUM | `active` / `ended` |

---

## Full API Reference

Base URL: `http://localhost:8000`

### `POST /auth/register` — `201`
Creates a guardian + blind user pair in one call.
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
    "emergency_phone": "+250780000000",
    "language": "en",
    "voice_speed": "Normal"
  }
}
```

---

### `POST /auth/login` — `200`
```json
{ "phone": "+250781000001", "password": "guardian123" }
```
Returns `token` + full user object. Guardian response includes nested `blind_user`.

**Error `401`:** `{ "detail": "Wrong phone or password." }`
**Error `422`:** Bilingual validation message (phone format or password too short)

---

### `POST /detect` — `200`
Sends a camera frame, returns detected objects in frontend format.

**Request:** `multipart/form-data` with field `file` (image).
**Optional header:** `Authorization: Bearer <token>` — if blind user token provided, danger alerts are auto-saved to the `alerts` table.

**Response:**
```json
{
  "objects": [
    {
      "name": "car",
      "isMoving": true,
      "distanceMeters": 1.0,
      "direction": "center",
      "dangerLevel": "danger"
    }
  ],
  "summary": "moving car, 1.0 meters ahead",
  "topDanger": "danger"
}
```

**Danger logic:**
- `car` or `stair` within 1.5m → always `danger`
- High danger objects (car, truck, bus...) within 5m → `danger`
- Medium danger objects (person, bicycle...) within 1.5m → `danger`, within 4m → `warning`

---

### `GET /guardian/dashboard` — `200`
**Auth:** Bearer token (guardian role)

Returns guardian info, linked blind user, last 10 alerts, active session.
Uses privacy-safe subquery — guardian only sees their own linked user's data.

```json
{
  "guardian": { "id": 2, "name": "Sarah Kamau", ... },
  "blind_user": { "id": 3, "name": "James Kamau", ... },
  "recent_alerts": [],
  "active_session": {
    "id": 1,
    "blind_id": 3,
    "started_at": "2026-05-09T17:12:53",
    "ended_at": null,
    "status": "active"
  }
}
```

---

### `POST /session/start` — `201`
**Auth:** Bearer token (blind role)
Starts a new navigation session. Auto-ends any previously active session.

### `POST /session/end` — `200`
**Auth:** Bearer token (blind role)
Ends the current active session, sets `ended_at` timestamp.

### `GET /session/active` — `200`
**Auth:** Bearer token (blind or guardian)
Returns the active session or `null`.

### `GET /session/history` — `200`
**Auth:** Bearer token (admin only). Returns last 100 sessions.

---

### `GET /alerts` — `200`
**Auth:** Bearer token (guardian or admin)
- Guardian → sees only their linked blind user's alerts
- Admin → sees all alerts

### `GET /alerts/{blind_id}` — `200`
**Auth:** Bearer token (admin only). All alerts for one blind user.

---

### `GET /users` — `200`
**Auth:** Bearer token (admin only). All users.

### `GET /users/{user_id}` — `200`
**Auth:** Bearer token (admin only). One user.

### `PATCH /users/{user_id}/status?status=inactive` — `200`
**Auth:** Bearer token (admin only). Activate or deactivate a user.

---

## Phone Number Validation

Accepted formats:
```
+250780000000   international MTN
+250790000000   international Airtel
+250720000000   international MTN
+250730000000   international Airtel
0780000000      local 10 digits
0790000000      local 10 digits
0720000000      local 10 digits
0730000000      local 10 digits
```

Rejected:
```
0711234567      invalid prefix (71)
078123456       too short (9 digits)
07812345678     too long (11 digits)
0788000000      ✅ valid (prefix 78, 10 digits)
```

Error message (bilingual):
```
Inomero ya telefoni ntabwo ari yo / Phone number is invalid.
Use +25078XXXXXXX or 078XXXXXXX (10 digits)
```

---

## Detection Engine

### Two models run on every frame
| Model | File | imgsz | Classes |
|---|---|---|---|
| Custom (indoor) | `yolov8n.onnx` | 320 | bed, door, sofa, stair, table, toilet, person |
| General (COCO) | `yolov8n.pt` | 320 | 80 classes — cars, people, animals, etc. |

> imgsz reduced from 640 to 320 — doubles CPU inference speed (~50–150ms per frame)

### Anti-chatter cooldown (per object+position)
| Priority | Cooldown | Reason |
|---|---|---|
| HIGH (car, stair) | 0s — always speaks | Safety critical |
| MEDIUM (person, chair) | 3s per position | Prevents "person right... person right..." |
| LOW (sink, bottle) | 8s per position | Background objects |

### Danger → Database pipeline
When a STRONG vibration is triggered (HIGH priority object detected):
1. `speak()` fires the voice alert immediately
2. `post_alert()` runs in a background thread
3. Alert is written to `alerts` table in PostgreSQL
4. Guardian sees it in real time via `GET /guardian/dashboard`

### Crowd detection
When 3+ people are detected within ~2m:
- Individual "person ahead" alerts are collapsed
- Single summary: `"Crowd ahead, navigate right. 3 people within 2 meters."`
- HIGH hazards (car, stair) always bypass crowd mode

### Path clear rule
Before announcing "Path clear":
- Must have 10 consecutive clean frames (no HIGH/MEDIUM objects)
- AND last alert in DB must be 5+ seconds ago
- Prevents false "path clear" announcements in busy areas

---

## Dependencies

```
fastapi
uvicorn
python-multipart
ultralytics
opencv-python
onnxruntime
sqlalchemy
bcrypt
python-jose[cryptography]
psycopg2-binary
```

---

## Authentication Flow

```
1. Frontend calls POST /auth/login
2. Backend verifies phone + bcrypt password
3. Returns JWT token (expires in 30 days)
4. Frontend stores token
5. Every protected request sends: Authorization: Bearer <token>
6. Backend decodes token → finds user → checks role
7. Wrong role → 403 Access denied
```

---

## Known Issues

| Issue | Fix |
|---|---|
| `libxcb.so.1` error on Render | Use `opencv-python-headless` in Docker |
| ONNX model task warning on load | Expected — model still runs correctly |
| Low detection confidence | `conf=0.2` set intentionally for 10-epoch model |
| Port 10048 already in use | Run: `netstat -aon \| findstr :8000` then `taskkill /PID <id> /F` |

---

## Roadmap

### Phase 3 — Frontend Connection (Next)
- Login screen → `POST /auth/login`
- Blind user camera screen → `POST /detect` with Bearer token
- Guardian screen → `GET /guardian/dashboard` (poll every 5s)
- Admin screen → `GET /users` + `GET /alerts`
- Session auto-start when blind user opens navigation screen

### Phase 4 — Production Deployment
- Deploy to Render / Railway with PostgreSQL add-on
- Increase YOLO training epochs (25–50) for better accuracy
- Add real distance measurement (depth camera or stereo vision)
- Wire vibration patterns to Android Vibrator API
- Add push notifications to guardian on danger alerts
