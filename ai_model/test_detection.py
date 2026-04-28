import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model 
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("model/labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
labels.insert(0, "???")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize image to model input size
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    image_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize if needed
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > 0.5:
            class_id = int(classes[i])
            label = labels[class_id] if class_id < len(labels) else "Unknown"
            print(f"Detected: {label} ({scores[i]:.2f})")

    # Show camera
    cv2.imshow("E-mboni Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()