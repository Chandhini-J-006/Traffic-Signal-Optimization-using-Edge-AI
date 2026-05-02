# Traffic Signal Optimization using Edge AI

---

## 📌 Overview
This project demonstrates real-time traffic signal optimization using Edge AI with TensorFlow Lite models. By deploying lightweight models directly on edge devices, the system reduces latency, avoids cloud dependency, and ensures efficient traffic management. Vehicle detection is performed on live video feeds, and traffic density is used to dynamically adjust signal timings.

---

## ⚙️ Features

- ✅ Model size check (must be under 50 MB for edge deployment)
- ⚡ Real-time inference with latency target < 100 ms
- 🚗 Vehicle detection (cars, buses, trucks, motorcycles)
- 📊 Weighted traffic density calculation for realism
- ⏱️ Adaptive signal control (20s, 40s, 60s green light)
- 🔋 Power efficiency via frame skipping

---

## 🛠️ Requirements
- Python 3.8+
- TensorFlow Lite
- OpenCV
- NumPy

---

### Install dependencies:

bash
```pip install tensorflow opencv-python numpy ```

---

## 📂 Project Structure

#### Code

├── model.tflite            (Pre-trained TensorFlow Lite model)

├── labelmap.txt            (Class labels)

├─  test1_video.mp4         (Sample traffic video 1)

├── test2_video.mp4         (Sample traffic video 2)

├── traffic.py              (Main script)

└── README.md               (Documentation)

---

## 🚦 Signal Logic

High Traffic (density > 10) → Green = 60 sec

Medium Traffic (density > 5) → Green = 40 sec

Low Traffic (density ≤ 5) → Green = 20 sec

---

## ▶️ Usage

Run the script:

bash
```python traffic.py```

---

## Result:

- Bounding boxes around detected vehicles.

- Density score and traffic decision displayed on video.

- Latency monitoring for real-time performance.

---

## 📊 Example Output

Vehicles counted: 12

Density: 28

Decision: HIGH TRAFFIC → Green = 60 sec

Latency: 85 ms → Real-time OK

---

## 🌍 Applications

- Smart city traffic management

- IoT-enabled intersections

- Edge AI deployment for low-latency systems
