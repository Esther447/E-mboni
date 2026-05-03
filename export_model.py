import shutil
import os

src = os.path.join("runs", "detect", "train", "weights", "best.onnx")
dst = "yolov8n.onnx"

if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"SUCCESS: Model copied to {dst} — ready to run yolo_detection.py")
else:
    print("ERROR: Training not complete yet. Run python train_model.py first.")
