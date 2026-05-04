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

WARNING_OBJECTS = [
    # Indoor navigation
    "door", "bed", "toilet", "sofa", "table", "chair",
    # Daily utility
    "bottle", "cup", "laptop", "cell phone",
    # Surface dangers (detectable via general model)
    "fire hydrant", "parking meter",
]

eye_of_blind_list = DANGER_OBJECTS + WARNING_OBJECTS

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
            
            if box_area > 120000:
                zone = "DANGER"
            elif box_area > 60000:
                zone = "WARNING"
            else:
                zone = "SAFE"

            # Horizontal position logic
            norm_x = ((x1 + x2) / 2) / w
            if norm_x < 0.33: pos = "on your left"
            elif norm_x < 0.67: pos = "straight ahead"
            else: pos = "on your right"

            # Priority: DANGER objects always override
            priority = "HIGH" if label in DANGER_OBJECTS and zone == "DANGER" else "NORMAL"

            if label not in detected_objects:
                detected_objects[label] = {"pos": pos, "zone": zone, "priority": priority}
            
            # Print vibration feedback
            if zone == "DANGER": print(f"📳📳📳 STRONG | {label} {pos}")
            elif zone == "WARNING": print(f"📳📳 MEDIUM | {label} {pos}")

    # Speak findings — HIGH priority bypasses cooldown
    now = time.time()
    high_priority = [d for d in detected_objects.values() if d["priority"] == "HIGH"]
    normal = [d for d in detected_objects.values() if d["priority"] == "NORMAL" and d["zone"] != "SAFE"]

    if high_priority:
        msg = "STOP. " + ", ".join(f"{l} {d['pos']}" for l, d in detected_objects.items() if d["priority"] == "HIGH")
        speak(msg)
        last_spoken_time = now
    elif normal and (now - last_spoken_time) > 3:
        msg = ", ".join(f"{l} {d['pos']}" for l, d in detected_objects.items() if d["priority"] == "NORMAL" and d["zone"] != "SAFE")
        print(f"AI Voice: {msg}")
        speak(msg)
        last_spoken_time = now

    cv2.imshow("E-mboni AI Engine", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()