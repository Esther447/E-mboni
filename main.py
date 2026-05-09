from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn
from roles import Role, can_access
from spatial_engine import RawDetection, process_detections, ALL_OBJECTS
from events import event_store

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

custom_model = YOLO("yolov8n.onnx")
general_model = YOLO("yolov8n.pt")


# --- RESPONSE MODEL ---
class Detection(BaseModel):
    object: str
    direction: str
    distance: Optional[str]
    priority: str
    vibe: Optional[str]
    vibe_pattern: Optional[str]
    vertical_zone: Optional[str]
    speech: str

class DetectResponse(BaseModel):
    detections: list[Detection]


# --- DETECT ENDPOINT ---
@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...), x_role: str = Header(default="user")):
    """
    Receives an image frame, runs both models, merges results,
    and returns the single highest-priority detection.

    Priority order: HIGH (stair, person, car...) → MEDIUM (door, chair...) → LOW (bottle, phone...)
    Role required: user (guardian/admin will receive 403)
    """
    try:
        role = Role(x_role.lower())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid role")

    if not can_access(role, "realtime_ai"):
        raise HTTPException(status_code=403, detail=f"Role '{role}' cannot access real-time detection")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    h, w = img.shape[:2]

    # Run both models and merge results
    results_c = custom_model(img, conf=0.2)   # custom: 7 indoor classes
    results_g = general_model(img, conf=0.25) # general: 80 COCO classes

    raw_detections = []
    seen_labels = set()

    for r in (results_c + results_g):
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            if label not in ALL_OBJECTS or label in seen_labels:
                continue
            x1, y1, x2, y2 = box.xyxy[0]
            raw_detections.append(RawDetection(
                label=label,
                box_area=float((x2 - x1) * (y2 - y1)),
                norm_x=float(((x1 + x2) / 2) / w),
                norm_y=float(((y1 + y2) / 2) / h),
            ))
            seen_labels.add(label)

    # Process, sort by priority, return top 1
    results = process_detections(raw_detections)

    # Log HIGH priority detections as safety events — frame is NOT stored
    for r in results:
        if r.priority == "HIGH":
            event_store.log_safety_event(
                hazard=r.object,
                direction=r.direction,
                distance=r.distance
            )
            break  # log only the top hazard per frame

    return DetectResponse(detections=[
        Detection(
            object=r.object,
            direction=r.direction,
            distance=r.distance,
            priority=r.priority,
            vibe=r.vibe,
            vibe_pattern=r.vibe_pattern,
            vertical_zone=r.vertical_zone,
            speech=r.speech,
        ) for r in results[:1]
    ])


# --- GUARDIAN ENDPOINTS ---
@app.get("/alerts")
async def get_alerts(x_role: str = Header(default="guardian"),
                     since: float = 0.0):
    try:
        role = Role(x_role.lower())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid role")
    if not can_access(role, "emergency_alerts"):
        raise HTTPException(status_code=403, detail="Access denied")
    alerts = event_store.get_guardian_alerts(since=since)
    return {"alerts": alerts, "note": "Text alerts only. No camera or location data."}


@app.get("/location/last-known")
async def last_known_location(user_id: str,
                               x_role: str = Header(default="guardian")):
    try:
        role = Role(x_role.lower())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid role")
    if not can_access(role, "last_known_location"):
        raise HTTPException(status_code=403, detail="Access denied")
    location = event_store.get_last_known_location(user_id)
    return {
        "location": location,
        "sharing_enabled": location is not None,
        "note": "Only available if user has enabled location sharing."
    }


# --- USER ENDPOINTS ---
@app.post("/location/share")
async def toggle_location_sharing(user_id: str, enabled: bool,
                                   x_role: str = Header(default="user")):
    try:
        role = Role(x_role.lower())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid role")
    if not can_access(role, "local_settings"):
        raise HTTPException(status_code=403, detail="Access denied")
    event_store.enable_location_sharing(user_id, enabled)
    return {"sharing_enabled": enabled}


# --- ADMIN ENDPOINTS ---
@app.get("/devices/status")
async def device_status(device_id: str,
                         x_role: str = Header(default="admin")):
    try:
        role = Role(x_role.lower())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid role")
    if not can_access(role, "device_management"):
        raise HTTPException(status_code=403, detail="Access denied")
    status = event_store.get_device_status(device_id)
    return {"device_id": device_id, "status": status,
            "note": "Device status only. No user data accessible."}


@app.post("/devices/ping")
async def device_ping(device_id: str, x_role: str = Header(default="user")):
    """Heartbeat endpoint — keeps device marked as online."""
    event_store.ping(device_id)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
