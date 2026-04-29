from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

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

            if label == "person":
                danger = "human"
            elif label in ["car", "motorcycle", "bicycle"]:
                danger = "vehicle"
            else:
                danger = "obstacle"

            print(f"{distance} obstacle {direction} ({label})")

            if distance in ["VERY CLOSE", "CLOSE"] and direction == "CENTER":
                print("⚠️ Obstacle ahead!")

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()