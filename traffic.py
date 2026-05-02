import cv2
import numpy as np
import time
import os
from tensorflow.lite.python.interpreter import Interpreter

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "model.tflite"
LABEL_PATH = "labelmap.txt"
VIDEO_PATH = "test2_video.mp4"

CONF_THRESHOLD = 0.25   # Lower = more detections
FRAME_SKIP = 2
MAX_MODEL_MB = 50
LATENCY_TARGET = 100

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# ==============================
# MODEL SIZE CHECK
# ==============================
model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Model Size: {model_size:.2f} MB")

if model_size > MAX_MODEL_MB:
    print("ERROR: Model too large")
    exit()

# ==============================
# LOAD MODEL
# ==============================
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully")

# ==============================
# LOAD LABELS
# ==============================
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ==============================
# VIDEO INPUT
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

frame_count = 0

# ==============================
# IMPROVED DENSITY FUNCTION
# ==============================
def get_density(classes, scores):
    density = 0
    vehicle_count = 0

    for i in range(len(scores)):
        if scores[i] > CONF_THRESHOLD:
            label = classes[i]

            if label in VEHICLE_CLASSES:
                vehicle_count += 1

                # Increased weights for realism
                if label == "car":
                    density += 2
                elif label in ["bus", "truck"]:
                    density += 5
                elif label == "motorcycle":
                    density += 1

    print(f"Vehicles counted: {vehicle_count}")
    return density


# ==============================
# UPDATED SIGNAL LOGIC
# ==============================
def decide_signal(density):
    if density > 10:
        return "HIGH TRAFFIC -> Green = 60 sec"
    elif density > 5:
        return "MEDIUM TRAFFIC -> Green = 40 sec"
    else:
        return "LOW TRAFFIC -> Green = 20 sec"


# ==============================
# MAIN LOOP
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames (power efficiency)
    if frame_count % FRAME_SKIP != 0:
        continue

    input_shape = input_details[0]['shape']
    h, w = input_shape[1], input_shape[2]

    img = cv2.resize(frame, (w, h))
    input_data = np.expand_dims(img, axis=0).astype(np.uint8)

    # ==============================
    # INFERENCE
    # ==============================
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # ==============================
    # LABEL MAPPING
    # ==============================
    detected_labels = []

    for c in classes:
        class_id = int(c)
        if class_id < len(labels):
            detected_labels.append(labels[class_id])
        else:
            detected_labels.append("unknown")

    # ==============================
    # DENSITY + SIGNAL
    # ==============================
    density = get_density(detected_labels, scores)
    decision = decide_signal(density)

    latency = (time.time() - start_time) * 1000

    if latency > LATENCY_TARGET:
        latency_status = "Latency > 100ms"
    else:
        latency_status = "Real-time OK"

    # ==============================
    # DRAW BOUNDING BOXES (VEHICLES ONLY)
    # ==============================
    imH, imW, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] > CONF_THRESHOLD and detected_labels[i] in VEHICLE_CLASSES:

            ymin = int(max(1, boxes[i][0] * imH))
            xmin = int(max(1, boxes[i][1] * imW))
            ymax = int(min(imH, boxes[i][2] * imH))
            xmax = int(min(imW, boxes[i][3] * imW))

            label = detected_labels[i]
            confidence = int(scores[i] * 100)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            text = f"{label}: {confidence}%"
            cv2.putText(frame, text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ==============================
    # DISPLAY TEXT
    # ==============================
    cv2.putText(frame, f"Density: {density:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, decision, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Latency: {latency:.2f} ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, latency_status, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Traffic Edge AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()