from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time
from spatial_engine import (
    RawDetection, process_detections, ALL_OBJECTS
)
from memory import MemoryEngine, MotionState, CrowdDetector, ConsistencyFilter

# --- CONFIG ---
IMGSZ_ONNX = 640  # ONNX model is fixed at 640x640 (exported size)
IMGSZ_PT   = 320  # .pt model is flexible — use 320 for speed on CPU
BLIND_ID = 3  # ID of the blind user in the DB (James Kamau = 3)

# --- VOICE ENGINE ---
engine = pyttsx3.init()
engine.setProperty('rate', 180)
is_speaking = False

def speak(text):
    global is_speaking
    if is_speaking: return
    def run():
        global is_speaking
        is_speaking = True
        engine.say(text)
        engine.runAndWait()
        is_speaking = False
    threading.Thread(target=run, daemon=True).start()

# --- ALERT POSTER ---
# Writes danger alerts directly to the database.
# Runs in a background thread so it never blocks the camera loop.
_alert_cooldown: dict[str, float] = {}  # prevent duplicate DB writes

def post_alert(object_name: str, message: str, level: str, blind_id: int = 3):
    """Insert a danger alert into PostgreSQL in a background thread."""
    now = time.time()
    key = f"{object_name}_{level}"
    # Only write once every 5 seconds per object+level to avoid DB spam
    if key in _alert_cooldown and (now - _alert_cooldown[key]) < 5.0:
        return
    _alert_cooldown[key] = now

    def _write():
        try:
            from database import SessionLocal, Alert, AlertLevelEnum
            db = SessionLocal()
            db.add(Alert(
                blind_id=blind_id,
                message=message,
                level=AlertLevelEnum(level),
            ))
            db.commit()
            db.close()
        except Exception as e:
            print(f"[alert DB error] {e}")

    threading.Thread(target=_write, daemon=True).start()


# --- MODELS ---
custom_model = YOLO("yolov8n.onnx", task="detect")
general_model = YOLO("yolov8n.pt",   task="detect")

eye_of_blind_list = ALL_OBJECTS


# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    speak("Camera not found. Please check your camera.")
    print("ERROR: Could not open camera.")
    exit(1)

speak("E-mboni started. Camera is ready.")
print("E-mboni AI Engine started. Camera opened.")

memory = MemoryEngine()
crowd = CrowdDetector()
consistency = ConsistencyFilter()
last_spoken_time = 0

# Per-object cooldown tracker — prevents "chatter" in busy environments
# Key: "object_direction" e.g. "person_right"
# Value: timestamp of last alert for that object+position
last_spoken: dict[str, float] = {}

COOLDOWN = {
    "HIGH":   0.0,   # HIGH danger always speaks — no cooldown
    "MEDIUM": 3.0,   # Navigation objects: 3 second cooldown
    "LOW":    8.0,   # Utility objects: 8 second cooldown
}


def should_alert(object_name: str, direction: str, priority: str) -> bool:
    """Returns True only if enough time has passed since we last alerted
    for this specific object+position combination."""
    key = f"{object_name}_{direction}"
    now = time.time()
    cooldown = COOLDOWN.get(priority, 3.0)
    if cooldown == 0.0:
        return True  # HIGH — always alert
    if key not in last_spoken or (now - last_spoken[key]) > cooldown:
        last_spoken[key] = now
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape

    results_c = custom_model(frame, imgsz=IMGSZ_ONNX, conf=0.6)
    results_g = general_model(frame, imgsz=IMGSZ_PT,   conf=0.6)

    # Collect raw detections
    raw_detections = []
    seen_labels = set()

    for r in (results_c + results_g):
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            if label not in eye_of_blind_list or label in seen_labels:
                continue
            x1, y1, x2, y2 = box.xyxy[0]
            raw_detections.append(RawDetection(
                label=label,
                box_area=float((x2 - x1) * (y2 - y1)),
                norm_x=float(((x1 + x2) / 2) / w),
                norm_y=float(((y1 + y2) / 2) / h),
                box_w=float(x2 - x1),
                box_h=float(y2 - y1),
            ))
            seen_labels.add(label)

    payload = process_detections(raw_detections)

    # --- CONSISTENCY FILTER: only keep objects seen 2+ frames in a row ---
    consistency.update([r.label for r in raw_detections])
    raw_detections = [r for r in raw_detections if consistency.is_confirmed(r.label)]
    payload = [p for p in payload if consistency.is_confirmed(p.object)]

    # --- VIBRATION LOCK: bottom zone ground hazards (Rwanda terrain) ---
    memory.update_vibe_lock(raw_detections)
    if memory.vibe_locked:
        print(f"📳📳📳 VIBE LOCK | {memory.vibe_lock_label} in bottom zone")

    # --- MOTION TRACKING: update area history every frame ---
    for raw in raw_detections:
        memory.track(raw.label, raw.norm_x, raw.norm_y, raw.box_area, "")

    # --- TEMPORAL FILTER + TTC + MOTION ESCALATION ---
    new_detections = []
    for p in payload:
        raw = next((r for r in raw_detections if r.label == p.object), None)
        if not raw:
            continue

        motion = memory.track(raw.label, raw.norm_x, raw.norm_y, raw.box_area, p.priority)

        # TTC: 20% growth in 2 frames — skip all polite logic, shout STOP immediately
        if memory.is_ttc_critical(raw.label):
            p.priority = "HIGH"
            p.vibe = "STRONG"
            p.speech = f"STOP! {p.object} approaching fast {p.direction}"
            new_detections.append(p)
            memory.remember(p.object, raw.norm_x, raw.norm_y, p.priority)
            continue

        # Escalate approaching objects to HIGH
        if motion == MotionState.APPROACHING:
            p.priority = "HIGH"
            p.vibe = "STRONG"
            p.speech = f"STOP. {p.object} approaching {p.direction}"
            if p.distance:
                p.speech += f", {p.distance}"

        # Suppress stationary/retreating known objects
        if motion in (MotionState.STATIONARY, MotionState.RETREATING):
            if memory.is_known(p.object, raw.norm_x, raw.norm_y):
                continue

        memory.remember(p.object, raw.norm_x, raw.norm_y, p.priority)
        new_detections.append(p)

    # --- STATE MONITOR: path clear trigger ---
    has_obstacles = any(p.priority in ["HIGH", "MEDIUM"] for p in payload)
    path_clear_msg = memory.check_path_clear(has_obstacles)

    # --- CROWD MODE CHECK (before individual alerts) ---
    is_crowd, crowd_msg = crowd.evaluate(raw_detections, payload)

    # Act on new detections only
    now = time.time()
    high   = [p for p in new_detections if p.priority == "HIGH"]
    medium = [p for p in new_detections if p.priority == "MEDIUM" and should_alert(p.object, p.direction, "MEDIUM")]
    low    = [p for p in new_detections if p.priority == "LOW"    and should_alert(p.object, p.direction, "LOW")]

    for p in new_detections:
        if p.vibe:
            print(f"📳 {p.vibe} | {p.object} {p.direction}")

    if high:
        # HIGH always bypasses crowd mode and cooldown
        alertable_high = [p for p in high if should_alert(p.object, p.direction, "HIGH")]
        if alertable_high:
            msg = "STOP. " + ", ".join(p.speech.replace("STOP. ", "") for p in alertable_high)
            speak(msg)
            last_spoken_time = now
            # --- POST TO DATABASE ---
            for p in alertable_high:
                print(f"🚨 DANGER logged → DB | {p.object} | {p.speech}")
                post_alert(
                    object_name=p.object,
                    message=p.speech,
                    level="danger",
                    blind_id=BLIND_ID,
                )
    elif is_crowd and (now - last_spoken_time) > 5:
        # Crowd summary replaces individual person alerts
        print(f"AI Voice [CROWD]: {crowd_msg}")
        speak(crowd_msg)
        last_spoken_time = now
    elif medium and not is_crowd:
        msg = ", ".join(p.speech for p in medium)
        print(f"AI Voice [NAV]: {msg}")
        speak(msg)
        last_spoken_time = now
    elif low and not is_crowd:
        msg = ", ".join(p.speech for p in low)
        print(f"AI Voice [UTIL]: {msg}")
        speak(msg)
        last_spoken_time = now
    elif path_clear_msg:
        print(f"AI Voice: {path_clear_msg}")
        speak(path_clear_msg)
        last_spoken_time = now

    try:
        cv2.imshow("E-mboni AI Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    except cv2.error:
        pass  # headless environment — skip display

cap.release()
cv2.destroyAllWindows()
