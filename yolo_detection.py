from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time

# --- INITIALIZE VOICE ---
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

# --- LOAD BOTH MODELS ---
# Custom: Stairs, Doors, etc.
custom_model = YOLO("yolov8n.onnx") 
# General: People, Bottles, Cups, Laptops, etc.
general_model = YOLO("yolov8n.pt") 

# Priority tiers for alert logic
DANGER_OBJECTS = [
    # Dynamic obstacles
    "person", "bicycle", "car", "bus", "truck", "motorcycle",
    # Drop-offs & elevation
    "stair",
    # Street hazards
    "traffic cone", "stop sign", "traffic light",
]

NAVIGATION_OBJECTS = [
    # Entry/exit points
    "door",
    # Large furniture
    "sofa", "bed", "dining table", "chair",
    # Path clearance
    "potted plant",
    # Appliance locators
    "refrigerator", "microwave", "oven", "sink", "toilet",
]

UTILITY_OBJECTS = [
    # Personal tech
    "laptop", "cell phone", "keyboard", "mouse", "remote",
    # Kitchen/dining
    "bottle", "cup", "bowl", "spoon", "fork", "knife",
    # Common items
    "backpack", "suitcase", "umbrella", "book", "handbag",
    # Animals (Rwanda environment)
    "dog", "cow", "bird", "cat", "horse",
    # Surface dangers
    "fire hydrant", "parking meter",
]

eye_of_blind_list = DANGER_OBJECTS + NAVIGATION_OBJECTS + UTILITY_OBJECTS

cap = cv2.VideoCapture(0)
last_spoken_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    
    # Running both models
    results_c = custom_model(frame, imgsz=640, conf=0.2)
    results_g = general_model(frame, imgsz=640, conf=0.25)

    detected_objects = {}

    for r in (results_c + results_g):
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            if label not in eye_of_blind_list: continue

            x1, y1, x2, y2 = box.xyxy[0]
            box_area = (x2 - x1) * (y2 - y1)

            # box_area thresholds:
            # > 120000 → ~< 1m  (VERY CLOSE)
            # > 60000  → ~< 2m  (CLOSE)
            # <= 60000 → FAR

            # Priority with distance gating per table
            if label in DANGER_OBJECTS:
                # HIGH: any distance — immediate voice + strong vibration
                priority = "HIGH"
                vibration = "📳📳📳 STRONG"
            elif label in NAVIGATION_OBJECTS and box_area > 60000:
                # MEDIUM: < 2m — voice direction + light vibration
                priority = "MEDIUM"
                vibration = "📳 LIGHT"
            elif label in UTILITY_OBJECTS and box_area > 120000:
                # LOW: < 1m — voice only, slow cooldown
                priority = "LOW"
                vibration = None
            else:
                priority = "NONE"
                vibration = None

            norm_x = ((x1 + x2) / 2) / w
            if norm_x < 0.33: pos = "on your left"
            elif norm_x < 0.67: pos = "straight ahead"
            else: pos = "on your right"

            if label not in detected_objects and priority != "NONE":
                detected_objects[label] = {"pos": pos, "priority": priority, "vibration": vibration}

            if vibration:
                print(f"{vibration} | {label} {pos}")

    now = time.time()
    high   = {l: d for l, d in detected_objects.items() if d["priority"] == "HIGH"}
    medium = {l: d for l, d in detected_objects.items() if d["priority"] == "MEDIUM"}
    low    = {l: d for l, d in detected_objects.items() if d["priority"] == "LOW"}

    if high:
        # Immediate voice + strong vibration — no cooldown
        msg = "STOP. " + ", ".join(f"{l} {d['pos']}" for l, d in high.items())
        speak(msg)
        last_spoken_time = now
    elif medium and (now - last_spoken_time) > 3:
        # Voice direction + light vibration — 3s cooldown
        msg = ", ".join(f"{l} {d['pos']}" for l, d in medium.items())
        print(f"AI Voice [NAV]: {msg}")
        speak(msg)
        last_spoken_time = now
    elif low and (now - last_spoken_time) > 8:
        # Voice only — slow 8s cooldown
        msg = ", ".join(f"{l} {d['pos']}" for l, d in low.items())
        print(f"AI Voice [UTIL]: {msg}")
        speak(msg)
        last_spoken_time = now

    cv2.imshow("E-mboni AI Engine", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()