from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

custom_model = YOLO("yolov8n.onnx")
general_model = YOLO("yolov8n.pt")

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


def get_direction(norm_x, norm_y):
    if norm_y < 0.3:
        vertical = "high"
    elif norm_y > 0.7:
        vertical = "low"
    else:
        vertical = None

    if norm_x < 0.2:    horizontal = "to your far left"
    elif norm_x < 0.4:  horizontal = "on your left"
    elif norm_x <= 0.6: horizontal = "straight ahead"
    elif norm_x <= 0.8: horizontal = "on your right"
    else:               horizontal = "to your far right"

    if vertical:
        return f"{horizontal}, {vertical}"
    return horizontal

def get_priority_and_vibe(label, box_area):
    if label in DANGER_OBJECTS:
        return "HIGH", "STRONG"
    elif label in NAVIGATION_OBJECTS and box_area > 60000:
        return "MEDIUM", "LIGHT"
    elif label in UTILITY_OBJECTS and box_area > 120000:
        return "LOW", None
    return "NONE", None


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    results_c = custom_model(img, conf=0.2)
    results_g = general_model(img, conf=0.25)

    payload = []
    seen_labels = set()

    for r in (results_c + results_g):
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            if label not in eye_of_blind_list or label in seen_labels:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            box_area = float((x2 - x1) * (y2 - y1))
            norm_x = float(((x1 + x2) / 2) / w)
            norm_y = float(((y1 + y2) / 2) / h)

            priority, vibe = get_priority_and_vibe(label, box_area)
            if priority == "NONE":
                continue

            direction = get_direction(norm_x, norm_y)

            if priority == "HIGH":
                speech = f"STOP. {label} {direction}"
            else:
                speech = f"{label} {direction}"

            payload.append({
                "object": label,
                "direction": direction,
                "priority": priority,
                "vibe": vibe,
                "speech": speech
            })
            seen_labels.add(label)

    # Sort by priority: HIGH first, then MEDIUM, then LOW
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    payload.sort(key=lambda x: priority_order.get(x["priority"], 3))

    # Return top detection only to keep voice clean
    return {"detections": payload[:1]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
