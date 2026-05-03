from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="navigation_data.yaml",
    epochs=20,
    imgsz=640,
    device="cpu"
)

path = model.export(format="onnx", imgsz=640)
print(f"SUCCESS: Your model is ready for the app at: {path}")
