# Esther — E-mboni Backend Specification
> Full backend contract for the E-mboni AI Mobility Assistant.
> Every detail here is derived directly from the existing frontend code.
> Build this backend and the frontend will connect without errors.

---

## 1. Project Overview

**App name:** E-mboni  
**Version:** 1.0.0  
**Purpose:** AI-powered assistive navigation for visually impaired users.  
**Frontend stack:** React Native (Expo), TypeScript, expo-router  
**Backend stack:** Python + FastAPI (already started in `E-mboni/main.py`)  
**Languages supported:** English (`en`) and Kinyarwanda (`rw`)

---

## 2. Architecture Summary

```
Mobile App (React Native)
        │
        │  HTTPS / REST JSON
        ▼
FastAPI Backend (Python)
        │
        ├── /auth/*         — login, register
        ├── /detect         — YOLOv8 object detection (already exists)
        ├── /users/*        — admin user management
        ├── /alerts/*       — guardian alert feed
        ├── /guardian/*     — guardian dashboard data
        └── /session/*      — blind user navigation sessions
```

---

## 3. User Roles

There are exactly **3 roles** in the system:

| Role       | Description                                              |
|------------|----------------------------------------------------------|
| `blind`    | The visually impaired user. Registered by a guardian.    |
| `guardian` | The caregiver/parent. Creates their own account + links a blind user. |
| `admin`    | System administrator. Has read access to all users, alerts, sessions. |

---

## 4. Database Models

Use any SQL database (PostgreSQL recommended) or SQLite for dev.

### 4.1 `users` table

| Column            | Type         | Notes                                      |
|-------------------|--------------|--------------------------------------------|
| `id`              | UUID / int   | Primary key                                |
| `name`            | string       | Full name                                  |
| `phone`           | string       | Unique. Format: `+250 7XX XXX XXX`         |
| `password_hash`   | string       | bcrypt hashed                              |
| `role`            | enum         | `blind` \| `guardian` \| `admin`           |
| `language`        | enum         | `en` \| `rw`                               |
| `voice_speed`     | enum         | `Slow` \| `Normal` \| `Fast`               |
| `status`          | enum         | `active` \| `inactive`                     |
| `guardian_id`     | FK → users   | Only set for `blind` users. Points to their guardian. |
| `emergency_phone` | string       | Optional. Only for `blind` users.          |
| `relationship`    | string       | Optional. Guardian's relationship to blind user (e.g. "Mother"). |
| `created_at`      | datetime     | Auto                                       |

### 4.2 `alerts` table

| Column       | Type       | Notes                                              |
|--------------|------------|----------------------------------------------------|
| `id`         | UUID / int | Primary key                                        |
| `blind_id`   | FK → users | The blind user who triggered the alert             |
| `message`    | string     | e.g. `"Moving car — 3m on the left"`               |
| `level`      | enum       | `safe` \| `warning` \| `danger`                    |
| `created_at` | datetime   | Auto                                               |

### 4.3 `sessions` table

| Column       | Type       | Notes                                              |
|--------------|------------|----------------------------------------------------|
| `id`         | UUID / int | Primary key                                        |
| `blind_id`   | FK → users | The blind user                                     |
| `started_at` | datetime   | When navigation started                            |
| `ended_at`   | datetime   | Nullable — null means session is active            |
| `status`     | enum       | `active` \| `ended`                                |

---

## 5. Authentication

### 5.1 `POST /auth/register`

Called from **guardian-register screen** then **blind-register screen** in sequence.

**Request body (JSON):**
```json
{
  "guardian": {
    "name": "Sarah Kamau",
    "phone": "+250 711 000 001",
    "password": "guardian123",
    "relationship": "Mother"
  },
  "blind_user": {
    "name": "James Kamau",
    "phone": "+250 711 000 002",
    "emergency_phone": "+250 700 000 000",
    "language": "en",
    "voice_speed": "Normal"
  }
}
```

**Response `201`:**
```json
{
  "guardian": {
    "id": 1,
    "name": "Sarah Kamau",
    "phone": "+250 711 000 001",
    "role": "guardian"
  },
  "blind_user": {
    "id": 2,
    "name": "James Kamau",
    "phone": "+250 711 000 002",
    "role": "blind",
    "guardian_id": 1
  },
  "token": "<jwt_token>"
}
```

**Validation errors `422`:**
```json
{ "detail": "Phone number already registered" }
```

---

### 5.2 `POST /auth/login`

Called from the **login screen**.

**Request body (JSON):**
```json
{
  "phone": "+250 711 000 001",
  "password": "guardian123"
}
```

**Response `200`:**
```json
{
  "token": "<jwt_token>",
  "user": {
    "id": 1,
    "name": "Sarah Kamau",
    "role": "guardian",
    "language": "en",
    "voice_speed": "Normal",
    "blind_user": {
      "id": 2,
      "name": "James Kamau",
      "status": "active",
      "language": "en"
    }
  }
}
```

For a `blind` user login, `blind_user` field is `null` and `role` is `"blind"`.  
For an `admin` login, `role` is `"admin"`, no `blind_user` field needed.

**Error `401`:**
```json
{ "detail": "Wrong phone or password." }
```

> The frontend currently uses demo accounts hardcoded. Replace them by wiring the login button to this endpoint.

**Demo accounts to seed in DB:**

| role     | phone              | password     | route after login |
|----------|--------------------|--------------|-------------------|
| guardian | +250 711 000 001   | guardian123  | /(guardian)       |
| blind    | +250 711 000 002   | blind123     | /(tabs)           |
| admin    | +250 711 000 000   | admin123     | /(admin)          |

---

## 6. Detection Endpoint

This already exists in `E-mboni/main.py`. The frontend calls it from `detectionService.ts`.

### 6.1 `POST /detect`

**Current frontend call** (in `detectionService.ts` — `detectFromFrame` function):
- Sends a base64 image string
- Expects back a `DetectionResult` object

**The frontend `DetectionResult` type:**
```typescript
interface DetectedObject {
  name: string;           // e.g. "car", "chair"
  isMoving: boolean;
  distanceMeters: number;
  direction: "left" | "center" | "right";
  dangerLevel: "safe" | "warning" | "danger";
}

interface DetectionResult {
  objects: DetectedObject[];
  summary: string;        // voice string, e.g. "moving car, 3.0 meters ahead"
  topDanger: "safe" | "warning" | "danger";
}
```

**The existing `/detect` endpoint returns:**
```json
{
  "detections": [
    {
      "object": "car",
      "direction": "on your left",
      "priority": "HIGH",
      "vibe": "STRONG",
      "speech": "STOP. car on your left"
    }
  ]
}
```

**⚠️ MISMATCH — You must update `/detect` to return the frontend format OR update `detectFromFrame` in the frontend.**

**Recommended: update `/detect` to return this format:**
```json
{
  "objects": [
    {
      "name": "car",
      "isMoving": true,
      "distanceMeters": 3.0,
      "direction": "left",
      "dangerLevel": "danger"
    }
  ],
  "summary": "moving car, 3.0 meters ahead",
  "topDanger": "danger"
}
```

**Direction mapping** (existing backend → frontend):
| Backend string      | Frontend value |
|---------------------|----------------|
| `"on your left"`    | `"left"`       |
| `"straight ahead"`  | `"center"`     |
| `"on your right"`   | `"right"`      |

**isMoving logic** — objects considered moving:
```
car, truck, bus, motorcycle, bicycle, person, dog, cat, vehicle, van, scooter, animal
```

**distanceMeters from box_area** (match frontend heuristic):
| box_area      | distanceMeters |
|---------------|----------------|
| > 0.4 (norm)  | 0.5            |
| > 0.2         | 1.0            |
| > 0.1         | 2.0            |
| > 0.05        | 3.5            |
| > 0.02        | 6.0            |
| > 0.01        | 10.0           |
| else          | 15.0           |

> Note: frontend normalizes bbox area as `width * height` where both are 0–1. Backend uses pixel area. Convert: `norm_area = box_area / (img_w * img_h)`.

**dangerLevel logic:**
```
HIGH_DANGER (car, truck, bus, motorcycle, vehicle, van):
  distance <= 5m → "danger"
  else           → "warning"

MEDIUM_DANGER (bicycle, scooter, person, dog, cat, animal, chair, table, bench, pole, fire hydrant, trash can, staircase, stairs, step):
  distance <= 1.5m → "danger"
  distance <= 4m   → "warning"
  else             → "safe"

Everything else:
  distance <= 1m → "warning"
  else           → "safe"
```

**summary string** — build in the language passed via query param `?lang=en` or `?lang=rw`:

English format: `"moving car, 3.0 meters ahead"` / `"chair, less than 1 meter ahead"`  
Kinyarwanda format: `"imodoka igenda, metero 3.0 imbere yawe"` / `"intebe, munsi ya metero imwe imbere yawe"`

Kinyarwanda object name map:
```
chair → intebe
car → imodoka
person → umuntu
table → ameza
dog → imbwa
stairs → inzitiro
bench → intebe ndefu
bus → bisi
pole → inkingi
bicycle → igare
trash can → agasanduku
truck → kamyo
motorcycle → moto
cat → injangwe
step → intambwe
```

Direction in Kinyarwanda:
```
center → imbere yawe
left   → ibumoso bwawe
right  → iburyo bwawe
```

**Request format** — the frontend sends base64 as JSON body:
```json
{ "image": "<base64_string>", "lang": "en" }
```

> The existing `/detect` uses `UploadFile`. You need to also accept JSON body with base64, OR update the frontend `detectFromFrame` to send multipart. Easiest: accept both.

---

## 7. Guardian Endpoints

All require `Authorization: Bearer <token>` header.

### 7.1 `GET /guardian/dashboard`

Returns the guardian's linked blind user status for the **Guardian Home screen**.

**Response `200`:**
```json
{
  "guardian": {
    "id": 1,
    "name": "Sarah Kamau",
    "initials": "SK",
    "phone": "+250 711 000 001"
  },
  "blind_user": {
    "id": 2,
    "name": "James Kamau",
    "initials": "JK",
    "status": "Scanning",
    "battery": 82,
    "last_seen": "2 min ago"
  },
  "recent_alerts": [
    {
      "message": "Chair — 1.5m ahead",
      "time": "5 min ago",
      "level": "warning"
    }
  ]
}
```

`status` values: `"Safe"` | `"Scanning"` | `"Navigating"` | `"Idle"` | `"Obstacle Detected"` | `"Danger Alert"`  
`battery` is an integer 0–100 (from device if available, else omit or return `null`).  
`last_seen` is a human-readable relative time string.

---

### 7.2 `GET /guardian/alerts`

Returns all alerts for the guardian's linked blind user. Used by **Alerts screen**.

**Response `200`:**
```json
{
  "blind_user_name": "James Kamau",
  "alerts": [
    {
      "id": 1,
      "message": "Chair detected — 1.5m ahead",
      "time": "Just now",
      "level": "warning"
    },
    {
      "id": 2,
      "message": "Moving car — 3m on the left",
      "time": "5 min ago",
      "level": "danger"
    }
  ]
}
```

`level` values: `"safe"` | `"warning"` | `"danger"`

---

### 7.3 `GET /guardian/tracking`

Returns the active session timeline for the guardian's linked blind user. Used by **Tracking screen**.

**Response `200`:**
```json
{
  "blind_user_name": "James Kamau",
  "session": {
    "status": "active",
    "duration_minutes": 8,
    "alert_count": 3,
    "safety_status": "Safe"
  },
  "timeline": [
    { "time": "10:40", "event": "Navigation stopped", "level": "muted" },
    { "time": "10:38", "event": "Path clear",          "level": "safe"  },
    { "time": "10:37", "event": "Moving car — 3m on the left", "level": "danger" },
    { "time": "10:35", "event": "Path clear",          "level": "safe"  },
    { "time": "10:34", "event": "Chair — 1.5m ahead",  "level": "warning" },
    { "time": "10:32", "event": "Navigation started",  "level": "safe"  }
  ]
}
```

`level` values: `"safe"` | `"warning"` | `"danger"` | `"muted"`

---

### 7.4 `GET /guardian/settings`

Returns the guardian's profile and linked blind user preferences. Used by **Settings screen**.

**Response `200`:**
```json
{
  "guardian": {
    "name": "Sarah Kamau",
    "initials": "SK",
    "phone": "+254 711 000 000"
  },
  "blind_user": {
    "name": "James Kamau",
    "language": "English",
    "voice_speed": "Normal"
  },
  "emergency_contacts": {
    "primary": "+254 700 000 000",
    "secondary": null
  },
  "notifications": {
    "push": true,
    "sms": false,
    "vibration": true
  }
}
```

---

### 7.5 `PATCH /guardian/settings`

Updates guardian notification preferences and blind user preferences.

**Request body:**
```json
{
  "notifications": {
    "push": true,
    "sms": false,
    "vibration": true
  },
  "blind_user": {
    "language": "en",
    "voice_speed": "Fast"
  }
}
```

**Response `200`:** same shape as `GET /guardian/settings`

---

## 8. Admin Endpoints

All require `Authorization: Bearer <token>` and `role == "admin"`.

### 8.1 `GET /admin/overview`

Used by **Admin Overview screen**.

**Response `200`:**
```json
{
  "stats": {
    "total_users": 128,
    "active_now": 14,
    "guardians": 76,
    "alerts_today": 32
  },
  "active_users": [
    { "name": "James Kamau",  "status": "Navigating", "initials": "JK" },
    { "name": "Alice Uwera",  "status": "Scanning",   "initials": "AU" },
    { "name": "Eric Mugisha", "status": "Idle",        "initials": "EM" }
  ],
  "recent_alerts": [
    { "user": "James Kamau",  "msg": "Moving car — 3m ahead",  "time": "2 min ago",  "level": "danger"  },
    { "user": "Alice Uwera",  "msg": "Chair detected — 1.5m",  "time": "8 min ago",  "level": "warning" },
    { "user": "Eric Mugisha", "msg": "Navigation started",      "time": "15 min ago", "level": "safe"    },
    { "user": "Grace Ineza",  "msg": "Stairs — 1m ahead",      "time": "22 min ago", "level": "danger"  }
  ]
}
```

---

### 8.2 `GET /admin/users`

Used by **Admin Users screen**.

**Response `200`:**
```json
{
  "total": 6,
  "users": [
    {
      "id": 1,
      "name": "James Kamau",
      "initials": "JK",
      "role": "blind",
      "guardian": "Sarah Kamau",
      "status": "active",
      "language": "EN"
    },
    {
      "id": 2,
      "name": "Sarah Kamau",
      "initials": "SK",
      "role": "guardian",
      "guardian": "—",
      "status": "active",
      "language": "EN"
    }
  ]
}
```

`role` values: `"blind"` | `"guardian"` | `"admin"`  
`status` values: `"active"` | `"inactive"`  
`language` values: `"EN"` | `"RW"`  
`guardian` for guardian accounts: `"—"` (em dash string)

---

## 9. Session Endpoints

### 9.1 `POST /session/start`

Called when blind user starts navigation (taps screen → `/(tabs)/navigation`).

**Request body:**
```json
{ "blind_user_id": 2 }
```

**Response `201`:**
```json
{ "session_id": 10, "started_at": "2025-01-15T10:32:00Z" }
```

---

### 9.2 `POST /session/end`

Called when blind user long-presses stop button → `/(tabs)/stop`.

**Request body:**
```json
{ "session_id": 10 }
```

**Response `200`:**
```json
{ "ended_at": "2025-01-15T10:40:00Z", "duration_minutes": 8 }
```

---

### 9.3 `POST /session/alert`

Called after each detection that produces a non-safe result. Frontend should call this from `detectFromFrame` after getting a result.

**Request body:**
```json
{
  "session_id": 10,
  "blind_user_id": 2,
  "message": "Moving car — 3.0 meters on your left",
  "level": "danger"
}
```

**Response `201`:**
```json
{ "alert_id": 55 }
```

---

## 10. Frontend Integration Points

### 10.1 Where to replace mock data

| File | Function/Variable | Replace with |
|------|-------------------|--------------|
| `services/detectionService.ts` | `detectFromFrame()` | Call `POST /detect` |
| `app/(auth)/login.tsx` | `DEMO_ACCOUNTS` + `handleLogin()` | Call `POST /auth/login` |
| `app/(auth)/guardian-register.tsx` | `handleNext()` | Call `POST /auth/register` (step 1) |
| `app/(auth)/blind-register.tsx` | `handleDone()` | Call `POST /auth/register` (step 2) |
| `app/(guardian)/index.tsx` | `USER` + `ALERTS` constants | Call `GET /guardian/dashboard` |
| `app/(guardian)/alerts.tsx` | `ALERTS` constant | Call `GET /guardian/alerts` |
| `app/(guardian)/tracking.tsx` | `ACTIVITY` constant | Call `GET /guardian/tracking` |
| `app/(guardian)/settings.tsx` | hardcoded profile values | Call `GET /guardian/settings` |
| `app/(admin)/index.tsx` | `STATS`, `RECENT_ALERTS`, `ACTIVE_USERS` | Call `GET /admin/overview` |
| `app/(admin)/users.tsx` | `USERS` constant | Call `GET /admin/users` |
| `app/(tabs)/navigation.tsx` | `startLoop()` | Call `POST /session/start` on mount, `POST /session/alert` on each detection, `POST /session/end` on stop |

---

### 10.2 Token storage

Store the JWT token in `AsyncStorage` with key `@emboni_token`.  
Read it and attach as `Authorization: Bearer <token>` header on all protected requests.

---

### 10.3 Base URL

Define a single constant in the frontend:
```typescript
// services/api.ts
export const API_BASE = 'http://<your-server-ip>:8000';
```

For local dev, use your laptop's LAN IP (e.g. `http://192.168.1.x:8000`).  
The mobile device and backend laptop must be on the same WiFi network.

---

## 11. Existing Backend File Reference

**`E-mboni/main.py`** — FastAPI app, already has `/detect` endpoint.  
**`E-mboni/requirements.txt`** — current deps: `fastapi`, `uvicorn`, `python-multipart`, `ultralytics`, `opencv-python-headless`, `onnxruntime`

**Add these dependencies:**
```
passlib[bcrypt]
python-jose[cryptography]
sqlalchemy
psycopg2-binary   # or use aiosqlite for SQLite
python-dotenv
```

**Run command:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 12. CORS

Already configured in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
This is fine for development. Restrict `allow_origins` to your app's domain in production.

---

## 13. Models Used

| Model file      | Type    | Classes | conf  | Used for                        |
|-----------------|---------|---------|-------|---------------------------------|
| `yolov8n.onnx`  | Custom  | 7       | 0.2   | Indoor: bed, door, sofa, stair, table, toilet, person |
| `yolov8n.pt`    | General | 80      | 0.25  | COCO: cars, people, animals, etc. |

Both run on every frame. Results are merged and filtered through `eye_of_blind_list`.

---

## 14. Color Reference (for any admin UI you build)

```
Background:  #0C0B12
Card:        #13111C
Border:      #1E1A2E
Text:        #EDE9F8
Muted:       #6B6480
Accent:      #A855F7
Safe:        #4ADE80
Warning:     #FBBF24
Danger:      #F87171
```

---

## 15. Summary Checklist

- [ ] `POST /auth/register` — creates guardian + blind user pair
- [ ] `POST /auth/login` — returns JWT + user object with role
- [ ] `POST /detect` — returns `DetectionResult` in frontend format
- [ ] `GET /guardian/dashboard` — blind user status + recent alerts
- [ ] `GET /guardian/alerts` — full alert list
- [ ] `GET /guardian/tracking` — session timeline
- [ ] `GET /guardian/settings` — profile + preferences
- [ ] `PATCH /guardian/settings` — update preferences
- [ ] `GET /admin/overview` — stats + active users + recent alerts
- [ ] `GET /admin/users` — full user list
- [ ] `POST /session/start` — start navigation session
- [ ] `POST /session/end` — end navigation session
- [ ] `POST /session/alert` — log a detection alert
- [ ] JWT middleware protecting all non-auth routes
- [ ] DB seeded with 3 demo accounts (guardian, blind, admin)
