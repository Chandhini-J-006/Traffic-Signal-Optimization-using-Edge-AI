# Traffic Signal Optimization using Edge AI

## 📌 Overview
This project demonstrates real-time traffic signal optimization using Edge AI with TensorFlow Lite models. By deploying lightweight models directly on edge devices, the system reduces latency, avoids cloud dependency, and ensures efficient traffic management. Vehicle detection is performed on live video feeds, and traffic density is used to dynamically adjust signal timings.

## ⚙️ Features

Model Size Check: Ensures .tflite model is under 50 MB for edge deployment.

Real-Time Inference: Achieves latency targets below 100 ms.

Vehicle Detection: Identifies cars, buses, trucks, and motorcycles using bounding boxes.

Traffic Density Calculation: Weighted scoring system for realistic traffic load estimation.

Adaptive Signal Control: Adjusts green light duration based on density (20s, 40s, 60s).

Power Efficiency: Frame skipping reduces computation load.

## 🛠️ Requirements

Python 3.8+

TensorFlow Lite

OpenCV

NumPy

### Install dependencies:

bash
pip install tensorflow opencv-python numpy

## 📂 Project Structure

#### Code

├── model.tflite            (Pre-trained TensorFlow Lite model)

├── labelmap.txt            (Class labels)

├─  test1_video.mp4         (Sample traffic video 1)

├── test2_video.mp4         (Sample traffic video 2)

├── traffic.py              (Main script)

└── README.md               (Documentation)

## 🚦 Signal Logic

High Traffic (density > 10) → Green = 60 sec

Medium Traffic (density > 5) → Green = 40 sec

Low Traffic (density ≤ 5) → Green = 20 sec

## ▶️ Usage

Run the script:

bash
python traffic.py

## Output:

Bounding boxes around detected vehicles.

Density score and traffic decision displayed on video.

Latency monitoring for real-time performance.

## 📊 Example Output

Vehicles counted: 12

Density: 28

Decision: HIGH TRAFFIC → Green = 60 sec

Latency: 85 ms → Real-time OK

## 🌍 Applications

Smart city traffic management

IoT-enabled intersections

Edge AI deployment for low-latency systems
