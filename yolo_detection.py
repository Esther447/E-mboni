from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time

engine = pyttsx3.init()
engine.setProperty('rate', 180)

is_speaking = False

def speak(text):
    global is_speaking
    if is_speaking:
        return
    def run():
        global is_speaking
        is_speaking = True
        engine.say(text)
        engine.runAndWait()
        is_speaking = False
    threading.Thread(target=run, daemon=True).start()

def vibrate(zone):
    if zone == "DANGER":
        print("📳📳📳 STRONG VIBRATION")
    elif zone == "WARNING":
        print("📳📳 MEDIUM VIBRATION")

custom_model = YOLO("yolov8n.onnx")
general_model = YOLO("yolov8n.pt")

road_obstacles = [
    "person", "car", "bottle", "laptop", "cell phone",
    "bed", "door", "sofa", "stair", "table", "toilet"
]

last_spoken = ""
last_spoken_time = 0
COOLDOWN_SECONDS = 3

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    
    # Run both models
    results_custom = custom_model(frame, imgsz=640, conf=0.2)
    results_general = general_model(frame, imgsz=640, conf=0.25)

    detected_objects = {}

    for r in (results_custom + results_general):
        for box in r.boxes:
            label = r.names[int(box.cls[0])]

            if label not in road_obstacles:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            box_area = (x2 - x1) * (y2 - y1)
            
            if box_area > 120000:
                zone = "DANGER"
            elif box_area > 50000:
                zone = "WARNING"
            else:
                zone = "SAFE"

            norm_x = ((x1 + x2) / 2) / w
            if norm_x < 0.33: direction = "on your left"
            elif norm_x < 0.67: direction = "straight ahead"
            else: direction = "on your right"

            if label not in detected_objects:
                detected_objects[label] = {"count": 1, "direction": direction, "zone": zone}
            else:
                detected_objects[label]["count"] += 1

    for label, data in detected_objects.items():
        direction = data["direction"]
        zone = data["zone"]

        print(f"{zone} | {label} {direction}")
        vibrate(zone)

        now = time.time()

        if zone == "DANGER":
            message = "STOP"
        elif zone == "WARNING":
            message = f"{label} {direction}"
        else:
            continue

        priority_labels = ["stair", "car"]
        is_priority = label in priority_labels and zone == "DANGER"

        if message != last_spoken or is_priority or (now - last_spoken_time) > COOLDOWN_SECONDS:
            speak(message)
            last_spoken = message
            last_spoken_time = now

    cv2.imshow("E-mboni AI Engine", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()