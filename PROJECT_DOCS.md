# E-mboni AI Engine — Project Documentation

## Overview
E-mboni is an AI-powered assistive navigation system for visually impaired users.
It uses a camera to detect objects in real time and provides:
- Voice alerts (direction + object type)
- Simulated vibration feedback (to be wired to Android in Phase 2)
- REST API for frontend (Flutter/Diane's UI) integration

---

## Project Structure

```
E-mboni/
├── yolo_detection.py         # Local real-time detection engine (webcam)
├── main.py                   # FastAPI server — REST API for frontend
├── train_model.py            # Fine-tunes YOLOv8 on custom dataset
├── export_model.py           # Copies best.onnx to root after training
├── navigation_data.yaml      # Dataset config used for training
├── data.yaml                 # Original Roboflow dataset config
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker config for Render deployment
├── yolov8n.onnx              # Custom trained model (7 indoor classes)
├── yolov8n.pt                # Pretrained general model (80 COCO classes)
├── train/                    # Training images + labels (from Roboflow)
├── valid/                    # Validation images + labels
├── test/                     # Test images + labels
└── runs/                     # Training output (weights, plots, metrics)
```

---

## Models

### 1. yolov8n.onnx (Custom Model)
- Fine-tuned on Roboflow Indoor Navigation dataset
- 10 epochs, CPU (i3), imgsz=640
- 7 classes: `bed, door, sofa, stair, table, toilet, person`
- conf=0.2 (low threshold for lightly trained model)

### 2. yolov8n.pt (General Model)
- Pretrained YOLOv8 Nano on COCO dataset
- 80 classes (people, cars, phones, animals, etc.)
- conf=0.25

Both models run on every frame. Results are merged and filtered through `eye_of_blind_list`.

---

## Detection Object List (eye_of_blind_list)

### HIGH Priority — DANGER (any distance → immediate alert)
| Category | Objects |
|---|---|
| Dynamic obstacles | person, bicycle, car, bus, truck, motorcycle |
| Drop-offs | stair |
| Street hazards | traffic cone, stop sign, traffic light |

### MEDIUM Priority — NAVIGATION (< 2m → voice + light vibration)
| Category | Objects |
|---|---|
| Entry/exit | door |
| Large furniture | sofa, bed, dining table, chair |
| Path clearance | potted plant |
| Appliances | refrigerator, microwave, oven, sink, toilet |

### LOW Priority — UTILITY (< 1m → voice only, slow cooldown)
| Category | Objects |
|---|---|
| Personal tech | laptop, cell phone, keyboard, mouse, remote |
| Kitchen/dining | bottle, cup, bowl, spoon, fork, knife |
| Common items | backpack, suitcase, umbrella, book, handbag |
| Animals (Rwanda) | dog, cow, bird, cat, horse |
| Surface dangers | fire hydrant, parking meter |

---

## Detection Logic

### Distance Estimation (box area proxy)
| box_area | Estimated Distance |
|---|---|
| > 120000 | ~< 1 metre (VERY CLOSE) |
| > 60000 | ~< 2 metres (CLOSE) |
| <= 60000 | FAR |

### Spatial Direction (normalized x-axis)
| norm_x range | Direction |
|---|---|
| 0.00 – 0.33 | on your left |
| 0.33 – 0.67 | straight ahead |
| 0.67 – 1.00 | on your right |

### Priority & Feedback Strategy
| Priority | Distance Gate | Voice | Vibration | Cooldown |
|---|---|---|---|---|
| HIGH | any | "STOP. [label] [direction]" | 📳📳📳 STRONG | none |
| MEDIUM | box_area > 60000 | "[label] [direction]" | 📳 LIGHT | 3 seconds |
| LOW | box_area > 120000 | "[label] [direction]" | none | 8 seconds |

---

## Helper Functions (Frontend-Ready)

```python
get_direction(norm_x)         # Returns: "on your left" / "straight ahead" / "on your right"
get_priority(label)           # Returns: "HIGH" / "MEDIUM" / "LOW" / "NONE"
process_detections(raw_list)  # Returns: JSON payload list
```

### JSON Payload Format (per detection)
```json
{
  "object": "person",
  "direction": "straight ahead",
  "priority": "HIGH",
  "vibe": "STRONG",
  "speech": "STOP. person straight ahead"
}
```

---

## REST API (main.py)

### Endpoint
```
POST /detect
```

### Request
- Content-Type: `multipart/form-data`
- Body: image file (jpg/png)

### Response
```json
{
  "detections": [
    {
      "object": "person",
      "direction": "straight ahead",
      "priority": "HIGH",
      "vibe": "STRONG",
      "speech": "STOP. person straight ahead"
    }
  ]
}
```

- Returns only the top 1 detection (highest priority) to keep voice clean
- CORS enabled for all origins — Diane's Flutter frontend can call this directly

---

## Voice System
- Engine: `pyttsx3` (offline, no internet needed)
- Speed: 180 wpm
- Non-blocking: runs in a background daemon thread
- Lock: `is_speaking` flag prevents overlapping speech

---

## Vibration System (Phase 1 — Simulated)
Printed to terminal as simulation. Phase 2 will wire to Android Vibrator API.

| Priority | Terminal Output |
|---|---|
| HIGH | 📳📳📳 STRONG |
| MEDIUM | 📳 LIGHT |
| LOW | (none) |

---

## Deployment

### Local (webcam detection)
```
python yolo_detection.py
```
- Uses full `opencv-python` for `cv2.imshow` support
- Press `q` to quit

### Server (Render / Docker)
```
docker build -t emboni .
docker run -p 8000:8000 emboni
```
- Uses `opencv-python-headless` (no GUI libs needed)
- Dockerfile installs: `libglib2.0-0, libsm6, libxext6, libxrender-dev, libxcb1`
- Port binding via `$PORT` env variable (Render compatible)

### API server locally
```
python main.py
# API available at http://localhost:8000/detect
```

---

## Dependencies

### requirements.txt
```
fastapi
uvicorn
python-multipart
ultralytics
opencv-python
onnxruntime
```

### Dockerfile overrides opencv with headless version for server

---

## Training Workflow

```
# Step 1 — Train the model
python train_model.py

# Step 2 — Copy best.onnx to root folder
python export_model.py

# Step 3 — Run detection
python yolo_detection.py
```

### Training Config (navigation_data.yaml)
- Dataset: Roboflow Indoor Navigation (Public Domain)
- Classes: bed, door, sofa, stair, table, toilet, person
- Epochs: 10 (hackathon demo)
- Device: CPU
- Export: ONNX format

---

## Git Branches
- `feature-ai-detection` — all AI development work
- `main` — stable/deployment branch (Render deploys from here)

---

## Known Issues & Fixes
| Issue | Fix |
|---|---|
| `libxcb.so.1` error on Render | Use `opencv-python-headless` in Docker |
| `cv2.imshow` crash on headless | Wrapped in `try/except` in `yolo_detection.py` |
| ONNX model task warning | Expected — model still runs correctly |
| Low detection confidence | conf=0.2 set for 10-epoch model |

---

## Phase 2 Roadmap
- Wire vibration to Android app via Bluetooth/WebSocket
- Increase training epochs (25–50) for better accuracy
- Add depth camera or stereo vision for real distance measurement
- Stream video frames from Flutter app to `/detect` endpoint
- Deploy Flutter mobile app with live camera feed
