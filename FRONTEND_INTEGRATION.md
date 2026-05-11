# E-mboni — Frontend Integration Guide
> For the React Native (Expo) frontend team.
> Read this entire file before writing a single line of code.

---

## Step 1 — Clone and Run the Backend

### 1.1 Clone the repo
```bash
git clone https://github.com/<your-username>/E-mboni.git
cd E-mboni
```

### 1.2 Install Python dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Set up PostgreSQL
You need PostgreSQL installed and running on your machine.

1. Create a database called `e-mboni`
2. Open `queries/create_tables.sql` in pgAdmin or any PostgreSQL client
3. Run the entire file — this creates all tables and seeds the 3 demo accounts

### 1.4 Set your database password
Open `database.py` and find this line:
```python
DATABASE_URL = "postgresql://postgres:Gervais0790194121  x@localhost:5432/e-mboni"
``` 

### 1.5 Start the backend server
```bash
python main.py
```

You should see:
```
INFO | E-mboni backend started. Database ready.
INFO | Uvicorn running on http://0.0.0.0:8000
```

### 1.6 Verify it works
Open your browser:
```
http://localhost:8000/docs
```
You will see the full Swagger UI with all endpoints.

---

## Step 2 — Base URL

In your React Native project, set the base URL once:

```typescript
// services/api.ts
const BASE_URL = "http://YOUR_COMPUTER_IP:8000";
// Example: "http://192.168.1.5:8000"
```

> ⚠️ Do NOT use `localhost` from the phone — the phone and computer must be on the same WiFi.
> Find your computer IP by running `ipconfig` in terminal and looking for IPv4 Address.

---

## Step 3 — Authentication Flow

### 3.1 Register (Guardian + Blind User together)

**Screen:** Guardian registration → Blind user registration → Done

```typescript
// POST /auth/register
const response = await fetch(`${BASE_URL}/auth/register`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    guardian: {
      name: "Sarah Kamau",
      phone: "+250781000001",
      password: "guardian123",
      relationship: "Mother"
    },
    blind_user: {
      name: "James Kamau",
      phone: "+250781000002",
      emergency_phone: "+250780000000",
      language: "en",
      voice_speed: "Normal"
    }
  })
});

const data = await response.json();
// data.token      ← save this JWT token
// data.guardian   ← guardian user object
// data.blind_user ← blind user object with guardian_id
```

---

### 3.2 Login

```typescript
// POST /auth/login
const response = await fetch(`${BASE_URL}/auth/login`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    phone: "+250781000001",
    password: "guardian123"
  })
});

const data = await response.json();
// Save the token to AsyncStorage
await AsyncStorage.setItem("token", data.token);
await AsyncStorage.setItem("user", JSON.stringify(data.user));

// Route based on role:
// data.user.role === "guardian" → navigate to /(guardian)
// data.user.role === "blind"    → navigate to /(tabs)
// data.user.role === "admin"    → navigate to /(admin)
```

**Response for guardian login:**
```json
{
  "token": "eyJ...",
  "user": {
    "id": 2,
    "name": "Sarah Kamau",
    "role": "guardian",
    "language": "en",
    "voice_speed": "Normal",
    "blind_user": {
      "id": 3,
      "name": "James Kamau",
      "status": "active",
      "language": "en"
    }
  }
}
```

**Response for blind login:** same but `blind_user` is `null` and `role` is `"blind"`.

---

### 3.3 Save and use the token

```typescript
// utils/auth.ts

export const getToken = async () => {
  return await AsyncStorage.getItem("token");
};

export const authHeader = async () => {
  const token = await getToken();
  return { "Authorization": `Bearer ${token}` };
};
```

Use it in every protected request:
```typescript
const headers = await authHeader();
const response = await fetch(`${BASE_URL}/guardian/dashboard`, { headers });
```

---

## Step 4 — Detection (Blind User Screen)

This is the most important endpoint. The blind user's camera sends frames here.

```typescript
// services/detectionService.ts

export const detectFromFrame = async (base64Image: string, token: string) => {
  // Convert base64 to blob
  const blob = await (await fetch(`data:image/jpeg;base64,${base64Image}`)).blob();

  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");

  const response = await fetch(`${BASE_URL}/detect`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      // Do NOT set Content-Type manually — FormData sets it automatically
    },
    body: formData,
  });

  return await response.json();
};
```

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

**What to do with the response:**
```typescript
const result = await detectFromFrame(frame, token);

// Speak the summary
if (result.summary) {
  Speech.speak(result.summary);
}

// Vibrate on danger
if (result.topDanger === "danger") {
  Vibration.vibrate([0, 500, 200, 500]); // pattern
} else if (result.topDanger === "warning") {
  Vibration.vibrate(200);
}
```

---

## Step 5 — Session Management (Blind User)

Call these when the blind user opens and closes the navigation screen.

```typescript
const headers = await authHeader();

// When navigation screen opens
await fetch(`${BASE_URL}/session/start`, {
  method: "POST",
  headers,
});

// When navigation screen closes
await fetch(`${BASE_URL}/session/end`, {
  method: "POST",
  headers,
});
```

---

## Step 6 — Guardian Dashboard

Poll this every 5 seconds to show live updates.

```typescript
// Poll every 5 seconds
useEffect(() => {
  const interval = setInterval(async () => {
    const headers = await authHeader();
    const response = await fetch(`${BASE_URL}/guardian/dashboard`, { headers });
    const data = await response.json();

    setBlindUser(data.blind_user);
    setAlerts(data.recent_alerts);      // last 10 alerts
    setActiveSession(data.active_session); // null if not navigating
  }, 5000);

  return () => clearInterval(interval);
}, []);
```

**Response:**
```json
{
  "guardian": { "id": 2, "name": "Sarah Kamau", ... },
  "blind_user": { "id": 3, "name": "James Kamau", "status": "active", ... },
  "recent_alerts": [
    {
      "id": 1,
      "message": "STOP. car straight ahead",
      "level": "danger",
      "created_at": "2026-05-09T17:12:53"
    }
  ],
  "active_session": {
    "id": 1,
    "started_at": "2026-05-09T17:12:53",
    "ended_at": null,
    "status": "active"
  }
}
```

---

## Step 7 — Admin Screen

```typescript
const headers = await authHeader();

// List all users
const users = await fetch(`${BASE_URL}/users`, { headers });

// List all alerts
const alerts = await fetch(`${BASE_URL}/alerts`, { headers });

// Deactivate a user
await fetch(`${BASE_URL}/users/3/status?status=inactive`, {
  method: "PATCH",
  headers,
});
```

---

## Step 8 — Error Handling

Every endpoint returns errors in this format:
```json
{ "detail": "Wrong phone or password." }
```

For validation errors (422):
```json
{
  "detail": [
    {
      "loc": ["body", "phone"],
      "msg": "Inomero ya telefoni ntabwo ari yo / Phone number is invalid."
    }
  ]
}
```

Handle them like this:
```typescript
const response = await fetch(`${BASE_URL}/auth/login`, { ... });

if (!response.ok) {
  const error = await response.json();

  if (Array.isArray(error.detail)) {
    // Validation error — show first message
    const msg = error.detail[0].msg.split(" / ").pop(); // get English part
    Alert.alert("Error", msg);
  } else {
    Alert.alert("Error", error.detail);
  }
  return;
}
```

---

## Demo Accounts for Testing

| Role | Phone | Password | Screen |
|---|---|---|---|
| admin | +250780000000 | admin123 | Admin dashboard |
| guardian | +250781000001 | guardian123 | Guardian dashboard |
| blind | +250781000002 | blind123 | Navigation screen |

---

## Phone Number Rules

The backend only accepts Rwandan phone numbers:
```
✅ +250780000000   international format
✅ 0780000000      local format (10 digits)
✅ Prefixes: 072, 073, 078, 079

❌ 0711234567      invalid prefix
❌ 078123456       too short
```

---

## Quick Reference — All Endpoints

| Method | Endpoint | Auth | Who uses it |
|---|---|---|---|
| POST | `/auth/register` | None | Registration screen |
| POST | `/auth/login` | None | Login screen |
| POST | `/detect` | Bearer (blind) | Navigation screen |
| POST | `/session/start` | Bearer (blind) | Navigation screen opens |
| POST | `/session/end` | Bearer (blind) | Navigation screen closes |
| GET | `/session/active` | Bearer (blind/guardian) | Check if navigating |
| GET | `/guardian/dashboard` | Bearer (guardian) | Guardian home screen |
| GET | `/alerts` | Bearer (guardian/admin) | Alert feed |
| GET | `/users` | Bearer (admin) | Admin user list |
| PATCH | `/users/{id}/status` | Bearer (admin) | Admin activate/deactivate |

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Connection refused` | Server not running | Run `python main.py` |
| `Connection refused` on phone | Using `localhost` | Use computer's IP address e.g. `192.168.1.5:8000` |
| `401 Unauthorized` | Token missing or expired | Login again and save new token |
| `403 Forbidden` | Wrong role for endpoint | Check you're using the right account |
| `422 Validation Error` | Wrong phone format or short password | Check phone is Rwandan format, password 6+ chars |
| `No tables found` in DB | Tables not created | Run `queries/create_tables.sql` in pgAdmin |
| Port 8000 in use | Old server still running | Run `netstat -aon \| findstr :8000` then `taskkill /PID <id> /F` |
