# 🚦 Smart Traffic Control using AI

An AI-powered Traffic Light Management System that dynamically controls traffic signals based on real-time vehicle detection using YOLOv5.

## 📌 Project Overview

This system:
- Accepts 4 lane images (A, B, C, D)
- Detects vehicles using YOLOv5
- Counts cars, buses, trucks, motorcycles
- Assigns green signal dynamically
- Adjusts signal timing based on vehicle density

## 🧠 How It Works

1. User selects 4 lane images.
2. YOLOv5 detects vehicles in each lane.
3. Vehicle count is calculated.
4. Lane with highest density gets GREEN signal.
5. Timer dynamically adjusts based on traffic load.
6. Signals rotate intelligently.

## 🛠 Technologies Used

- Python
- Tkinter (GUI)
- OpenCV
- PyTorch
- YOLOv5 (Ultralytics)
- NumPy
- Pillow

## 🚀 Features

- AI-based vehicle detection
- Dynamic signal timing
- Real-time signal switching
- Graphical user interface
- Smart density-based prioritization

## 📂 Project Structure

