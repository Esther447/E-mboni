import shutil
import os
import glob

# Find the latest training run
runs = sorted(glob.glob(os.path.join("runs", "detect", "train*", "weights", "best.onnx")))

if not runs:
    print("ERROR: No best.onnx found. Make sure training completed and ONNX was exported.")
else:
    src = runs[-1]  # Use the latest run
    dst = "yolov8n.onnx"
    shutil.copy(src, dst)
    print(f"SUCCESS: Model copied from {src} to {dst} — ready to run yolo_detection.py")
