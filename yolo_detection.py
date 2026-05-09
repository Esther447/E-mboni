from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time

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

# --- OBJECT TIERS ---
DANGER_OBJECTS = [
    "person", "bicycle", "car", "bus", "truck", "motorcycle",
    "stair", "traffic cone", "stop sign", "traffic light",
]

NAVIGATION_OBJECTS = [
    "door", "sofa", "bed", "dining table", "chair", "potted plant",
    "refrigerator", "microwave", "oven", "sink", "toilet",
]

UTILITY_OBJECTS = [
    "laptop", "cell phone", "keyboard", "mouse", "remote",
    "bottle", "cup", "bowl", "spoon", "fork", "knife",
    "backpack", "suitcase", "umbrella", "book", "handbag",
    "dog", "cow", "bird", "cat", "horse",
    "fire hydrant", "parking meter",
]

eye_of_blind_list = DANGER_OBJECTS + NAVIGATION_OBJECTS + UTILITY_OBJECTS


# --- HELPER FUNCTIONS ---
def get_direction(norm_x, norm_y):
    if norm_y < 0.3:
        vertical = "high"
    elif norm_y > 0.7:
        vertical = "low"
    else:
        vertical = None

    if norm_x < 0.2:   horizontal = "to your far left"
    elif norm_x < 0.4: horizontal = "on your left"
    elif norm_x <= 0.6: horizontal = "straight ahead"
    elif norm_x <= 0.8: horizontal = "on your right"
    else:               horizontal = "to your far right"

    if vertical:
        return f"{horizontal}, {vertical}"
    return horizontal

def get_priority(label):
    if label in DANGER_OBJECTS: return "HIGH"
    if label in NAVIGATION_OBJECTS: return "MEDIUM"
    if label in UTILITY_OBJECTS: return "LOW"
    return "NONE"

def process_detections(raw_detections):
    payload = []
    for det in raw_detections:
        direction = get_direction(det["norm_x"], det["norm_y"])
        priority = get_priority(det["label"])

        if priority == "HIGH":
            vibe = "STRONG"
            speech = f"STOP. {det['label']} {direction}"
        elif priority == "MEDIUM" and det["box_area"] > 60000:
            vibe = "LIGHT"
            speech = f"{det['label']} {direction}"
        elif priority == "LOW" and det["box_area"] > 120000:
            vibe = None
            speech = f"{det['label']} {direction}"
        else:
            continue

        payload.append({
            "object": det["label"],
            "side": direction,
            "priority": priority,
            "vibe": vibe,
            "speech": speech
        })
    return payload


# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
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
            raw_detections.append({
                "label": label,
                "box_area": float((x2 - x1) * (y2 - y1)),
                "norm_x": float(((x1 + x2) / 2) / w),
                "norm_y": float(((y1 + y2) / 2) / h),
            })
            seen_labels.add(label)

    # Process into structured payload
    payload = process_detections(raw_detections)

    # Act on payload
    now = time.time()
    high   = [p for p in payload if p["priority"] == "HIGH"]
    medium = [p for p in payload if p["priority"] == "MEDIUM"]
    low    = [p for p in payload if p["priority"] == "LOW"]

    for p in payload:
        if p["vibe"]:
            print(f"📳 {p['vibe']} | {p['object']} {p['side']}")

    if high:
        msg = "STOP. " + ", ".join(p["speech"].replace("STOP. ", "") for p in high)
        speak(msg)
        last_spoken_time = now
    elif medium and (now - last_spoken_time) > 3:
        msg = ", ".join(p["speech"] for p in medium)
        print(f"AI Voice [NAV]: {msg}")
        speak(msg)
        last_spoken_time = now
    elif low and (now - last_spoken_time) > 8:
        msg = ", ".join(p["speech"] for p in low)
        print(f"AI Voice [UTIL]: {msg}")
        speak(msg)
        last_spoken_time = now

    try:
        cv2.imshow("E-mboni AI Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    except cv2.error:
        pass  # headless environment — skip display

cap.release()
cv2.destroyAllWindows()
