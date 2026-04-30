from ultralytics import YOLO
import cv2
import pyttsx3
import threading

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

def vibrate(level):
    if level == "VERY CLOSE":
        print("📳📳📳 STRONG VIBRATION")
    elif level == "CLOSE":
        print("📳📳 MEDIUM VIBRATION")
    elif level == "MEDIUM":
        print("📳 LIGHT VIBRATION")

# Load YOLO model
model = YOLO("yolov8n.pt")

last_spoken = ""
cooldown = 0

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=320)

    h, w, _ = frame.shape
    road_obstacles = ["person", "car", "bicycle", "motorcycle", "chair"]

    detected_objects = {}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if label in road_obstacles and conf > 0.7:
                x1, y1, x2, y2 = box.xyxy[0]

                box_area = (x2 - x1) * (y2 - y1)

                if box_area > 100000:
                    distance = "VERY CLOSE"
                elif box_area > 50000:
                    distance = "CLOSE"
                elif box_area > 20000:
                    distance = "MEDIUM"
                else:
                    distance = "FAR"

                center_x = (x1 + x2) / 2

                if center_x < w / 3:
                    direction = "LEFT"
                elif center_x < 2 * w / 3:
                    direction = "CENTER"
                else:
                    direction = "RIGHT"

                if label not in detected_objects:
                    detected_objects[label] = {"count": 0, "direction": direction, "distance": distance}
                detected_objects[label]["count"] += 1
                detected_objects[label]["direction"] = direction
                detected_objects[label]["distance"] = distance

    for label, data in detected_objects.items():
        if data["count"] >= 3:
            direction = data["direction"]
            distance = data["distance"]

            print(f"{distance} obstacle {direction} ({label})")

            vibrate(distance)

            message = ""
            if distance in ["VERY CLOSE", "CLOSE"]:
                if direction == "CENTER":
                    message = f"{label} ahead"
                elif direction == "LEFT":
                    message = f"{label} on your left"
                elif direction == "RIGHT":
                    message = f"{label} on your right"

            if message and (message != last_spoken or cooldown > 10):
                print(message)
                speak(message)
                last_spoken = message
                cooldown = 0
            else:
                cooldown += 1

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()