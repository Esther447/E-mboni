from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="navigation_data.yaml",
    epochs=10,
    imgsz=640,
    device="cpu"
)

model.export(format="onnx")
