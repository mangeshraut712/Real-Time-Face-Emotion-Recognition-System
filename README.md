<div align="center">

# 🎭 Emotion AI: Real-Time Face Emotion Recognition

Real-time facial emotion analysis powered by a TensorFlow vision pipeline and a React dashboard.

![React](https://img.shields.io/badge/React-18.2-61dafb?style=flat&logo=react&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178c6?style=flat&logo=typescript&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-ff6f00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat&logo=opencv&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Stack](#stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
- [License](#license)

## Overview

Emotion AI combines local webcam inference, face detection, and emotion classification into a single experience. The project ships with a browser-based dashboard, a desktop GUI, and a training script so you can run the model, inspect predictions, or retrain the pipeline from the same codebase.

## Features

- Real-time emotion detection from webcam video with face overlays and confidence labels.
- Dual interfaces: a Flask-powered web app and a Tkinter-based desktop app.
- TensorFlow/Keras emotion classification with OpenCV and MediaPipe-based preprocessing.
- Session-friendly controls for camera toggling, exports, and visual feedback.
- Optional model retraining and local-first execution without external APIs.

## Stack

- Frontend: React 18, TypeScript, Vite, Tailwind CSS, Zustand, Recharts.
- Backend: Python 3.10+, Flask, OpenCV, MediaPipe, TensorFlow/Keras.
- Tooling: pytest, Ruff, MyPy, pre-commit, npm scripts for the web client.

## Quick Start

### Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- `pip` and `npm`

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd src/web/frontend
npm install
npm run build
cd ../../..
```

### Run

```bash
# Web app
python run_web.py

# Desktop GUI
python run_gui.py

# One-shot build + web launch
./scripts/start.sh

# Retrain the model
python train.py
```

The web interface starts on `http://localhost:5000` by default.

## Project Structure

```text
.
├── docs/                 # Research paper and sample screenshots
├── haarcascade_files/    # Face detection cascades
├── models/               # Trained emotion models
├── scripts/              # Launch and test helpers
├── src/
│   ├── core/             # Detection and inference logic
│   ├── gui/              # Desktop UI
│   └── web/              # Flask app and React frontend
├── tests/                # Unit and integration tests
├── run_gui.py            # Desktop entry point
├── run_web.py            # Web entry point
└── train.py              # Model training entry point
```

## Scripts

- `python run_web.py` starts the Flask web server.
- `python run_gui.py` opens the desktop emotion-recognition app.
- `python train.py` retrains the emotion model.
- `./scripts/start.sh` builds the frontend and launches the web server.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
