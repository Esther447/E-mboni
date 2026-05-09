from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time
from spatial_engine import (
    RawDetection, process_detections, ALL_OBJECTS
)
from memory import MemoryEngine, MotionState, CrowdDetector

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

# --- MODELS ---
custom_model = YOLO("yolov8n.onnx")
general_model = YOLO("yolov8n.pt")

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
last_spoken_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape

    results_c = custom_model(frame, imgsz=640, conf=0.2)
    results_g = general_model(frame, imgsz=640, conf=0.25)

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
            ))
            seen_labels.add(label)

    payload = process_detections(raw_detections)

    # --- MOTION TRACKING: update area history every frame ---
    for raw in raw_detections:
        memory.track(raw.label, raw.norm_x, raw.norm_y, raw.box_area, "")

    # --- TEMPORAL FILTER + MOTION ESCALATION ---
    new_detections = []
    for p in payload:
        raw = next((r for r in raw_detections if r.label == p.object), None)
        if not raw:
            continue

        motion = memory.track(raw.label, raw.norm_x, raw.norm_y, raw.box_area, p.priority)

        # Escalate approaching objects to HIGH regardless of original priority
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
    medium = [p for p in new_detections if p.priority == "MEDIUM"]
    low    = [p for p in new_detections if p.priority == "LOW"]

    for p in new_detections:
        if p.vibe:
            print(f"📳 {p.vibe} | {p.object} {p.direction}")

    if high:
        # HIGH always bypasses crowd mode
        msg = "STOP. " + ", ".join(p.speech.replace("STOP. ", "") for p in high)
        speak(msg)
        last_spoken_time = now
    elif is_crowd and (now - last_spoken_time) > 5:
        # Crowd summary replaces individual person alerts
        print(f"AI Voice [CROWD]: {crowd_msg}")
        speak(crowd_msg)
        last_spoken_time = now
    elif medium and not is_crowd and (now - last_spoken_time) > 3:
        msg = ", ".join(p.speech for p in medium)
        print(f"AI Voice [NAV]: {msg}")
        speak(msg)
        last_spoken_time = now
    elif low and not is_crowd and (now - last_spoken_time) > 8:
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
