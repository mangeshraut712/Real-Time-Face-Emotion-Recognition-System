# 🎭 Emotion AI: Real-Time Face Emotion Recognition (v2.0)

<div align="center">

![React](https://img.shields.io/badge/React-18.2-61dafb?style=flat&logo=react&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-5.0-646cff?style=flat&logo=vite&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178c6?style=flat&logo=typescript&logoColor=white)
![Zustand](https://img.shields.io/badge/Zustand-State-black?style=flat&logo=react&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-ff6f00?style=flat&logo=tensorflow&logoColor=white)

**Advanced Real-Time Emotion Analysis Powered by Deep Learning & Modern Web Technologies**

</div>

---

## 🚀 Overview

**Emotion AI** is a state-of-the-art emotion recognition system that leverages **Computer Vision** and **Deep Learning** to detect human emotions in real-time. Built with a privacy-first approach, it processes video feeds locally using a high-performance **Flask** backend and visualizes insights via a premium **React** frontend.

The system uses **MediaPipe** for face detection and a custom **TensorFlow/Keras** model for emotion classification, achieving high accuracy across 7 emotional states: *Angry, Disgust, Scared, Happy, Sad, Surprised, and Neutral*.

## ✨ Key Features

### 🎨 Modern Frontend Experience
- **Premium UI**: Designed with **Shadcn/UI**, **Radix Primitives**, and **Glassmorphism** aesthetics.
- **State Management**: Powered by **Zustand** for seamless global state sharing across components.
- **Data Visualization**: Real-time emotion probability charts using **Recharts**.
- **Smooth Animations**: Powered by **Framer Motion** for fluid transitions and layout shifts.
- **Responsive Layout**: Optimized single-page dashboard that adapts to all screen sizes.

### 🧠 Advanced AI Backend
- **Multi-Stage Pipeline**: 
  1. Face detection (MediaPipe Face Mesh)
  2. Region of Interest (ROI) extraction
  3. Emotion Classification (CNN Model)
- **High Performance**: Optimized for 30-60 FPS inference on standard CPUs.
- **Streaming Architecture**: Low-latency MJPEG video streaming via Flask.

### 🛠️ Utilities
- **Session Recording**: Capture emotion data sessions for analysis.
- **Export Data**: Download session history as **JSON** or **CSV** for external processing.
- **Keyboard Shortcuts**: Power-user controls for camera, sound, and export.
- **Audio Feedback**: Context-aware sound effects for interactions.

---

## 🏗️ Technology Stack

| Domain | Technology | Usage |
|:---:|:---|:---|
| **Frontend** | **React 18** + **Vite** | Core UI Framework |
| | **TypeScript** | Type Safety |
| | **Tailwind CSS** | Styling engine |
| | **Zustand** | Global State Management |
| | **TanStack Query** | Server State / API Caching |
| | **Framer Motion** | Animations |
| | **Recharts** | Data Visualization |
| **Backend** | **Python 3.10+** | Core Logic |
| | **Flask** | Web Server / API |
| | **TensorFlow** | Emotion Classification Model |
| | **OpenCV** | Image Processing |
| | **MediaPipe** | Face Detection |

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** v18+ 
- **Python** 3.10+
- **pip** & **npm**

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mangeshraut712/Real-Time-Face-Emotion-Recognition-System.git
   cd Real-Time-Face-Emotion-Recognition-System
   ```

2. **Setup Backend**
   ```bash
   # Create virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Setup Frontend**
   ```bash
   cd src/web/frontend
   npm install
   npm run build
   cd ../../..
   ```

4. **Run the Application**
   ```bash
   # Run via script (handles both backend and frontend serving)
   python run_web.py
   # OR
   ./scripts/start.sh
   ```

   Open your browser at `http://localhost:5000`.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Description |
|:---:|:---|
| `Ctrl + C` | Toggle Camera On/Off |
| `Ctrl + S` | Toggle Sound Feedback |
| `Ctrl + F` | Toggle Fullscreen |
| `Ctrl + E` | Export Session Data (JSON) |
| `Ctrl + X` | Clear Session History |

---

## 📂 Project Structure

```
├── src/
│   ├── core/               # Machine Learning Logic
│   │   ├── face_detector.py     # Face Detection (MediaPipe)
│   │   └── emotion_detector.py  # Emotion Prediction (TF)
│   ├── gui/                # Desktop Application (PyQt/Tkinter)
│   ├── web/
│   │   ├── app.py          # Flask Application Entry
│   │   └── frontend/       # React Application
│   │       ├── src/
│   │       │   ├── components/  # Dashboard, Header, Charts
│   │       │   ├── store.ts     # Zustand Store
│   │       │   └── lib/         # Utilities (API, Utils)
│   │       └── dist/            # Built Frontend Assets
├── models/                 # Pre-trained .h5 Models
├── scripts/                # Launch & Test Scripts
├── deployment/             # Docker Configuration
├── docs/                   # Documentation & Assets
└── tests/                  # Unit & Integration Tests
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>Elevating Human-Computer Interaction through Emotion AI</b>
</div>
