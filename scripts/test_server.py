#!/usr/bin/env python3
"""
Real AI Emotion Detection Server
Uses TensorFlow/Keras model + OpenCV for actual emotion detection.
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

# Try to load TensorFlow
try:
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  TensorFlow not found. Using random emotions.")

app = Flask(__name__)

static_folder = Path(__file__).parent.parent / "src/web/static"
models_folder = Path(__file__).parent.parent / "models"
haarcascade_folder = Path(__file__).parent.parent / "haarcascade_files"

# Ensure frontend build exists for static serving
if not static_folder.exists():
    raise RuntimeError(
        f"Static folder not found at {static_folder}. "
        "Build the frontend with: cd src/web/frontend && npm install && npm run build"
    )

# State
is_streaming = False
camera = None
camera_lock = threading.Lock()
current_result = None
face_cascade = None
emotion_model = None

# Emotions (same order as training)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


def load_models():
    """Load face cascade and emotion model."""
    global face_cascade, emotion_model

    # Load Haar cascade
    cascade_path = haarcascade_folder / "haarcascade_frontalface_default.xml"
    if cascade_path.exists():
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        print("✅ Haar cascade loaded")
    else:
        # Fallback to OpenCV's built-in
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("✅ Using OpenCV's built-in cascade")

    # Load emotion model
    if HAS_TF:
        model_path = models_folder / "mini_XCEPTION.02-0.39.keras"
        if not model_path.exists():
            # Try other extensions
            for ext in [".h5", ".hdf5", ".keras"]:
                alt_path = list(models_folder.glob(f"*{ext}"))
                if alt_path:
                    model_path = alt_path[0]
                    break

        if model_path.exists():
            try:
                emotion_model = tf.keras.models.load_model(str(model_path), compile=False)
                print(f"✅ Emotion model loaded: {model_path.name}")
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                emotion_model = None
        else:
            print("⚠️  No emotion model found in models/")


# Load models at startup
load_models()


def get_camera():
    """Get or create camera instance."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            time.sleep(0.3)
        return camera


def release_camera():
    """Release camera."""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


def detect_emotion(face_img):
    """Detect emotion from face image using AI model."""
    global current_result

    if emotion_model is None:
        # Fallback to random if no model
        import random

        emotion = random.choice(EMOTIONS)
        probs = {e: random.random() for e in EMOTIONS}
        total = sum(probs.values())
        probs = {k: round(v / total, 4) for k, v in probs.items()}
        return {"emotion": emotion, "confidence": probs[emotion], "probabilities": probs}

    try:
        # Preprocess
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype("float32") / 255.0
        # Match training preprocessing (v2)
        normalized = (normalized - 0.5) * 2.0

        # Reshape for model
        input_data = normalized.reshape(1, 48, 48, 1)

        # Predict
        predictions = emotion_model.predict(input_data, verbose=0)[0]

        # Get results
        emotion_idx = int(np.argmax(predictions))
        emotion = EMOTIONS[emotion_idx]
        confidence = float(predictions[emotion_idx])

        probabilities = {EMOTIONS[i]: round(float(predictions[i]), 4) for i in range(len(EMOTIONS))}

        return {"emotion": emotion, "confidence": confidence, "probabilities": probabilities}
    except Exception as e:
        print(f"Detection error: {e}")
        return None


def generate_frames():
    """Generate MJPEG frames from camera with face detection overlay."""
    global current_result
    try:
        while True:
            if not is_streaming:
                time.sleep(0.1)
                continue

            try:
                cam = get_camera()
                if cam is None or not cam.isOpened():
                    time.sleep(0.2)
                    continue

                with camera_lock:
                    ret, frame = cam.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                # Flip for mirror effect
                frame = cv2.flip(frame, 1)

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                if face_cascade is not None:
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(80, 80),
                    )

                    for x, y, w, h in faces:
                        # Detect emotion on face ROI
                        face_bgr = frame[y : y + h, x : x + w]
                        result = detect_emotion(face_bgr)
                        if result:
                            current_result = result
                            emotion = result["emotion"]
                            confidence = result["confidence"]

                            # Draw rectangle + label
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            label = f"{emotion.upper()} {confidence * 100:.0f}%"
                            cv2.putText(
                                frame,
                                label,
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2,
                            )

                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                frame_bytes = buffer.tobytes()

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            except Exception as e:
                print(f"Frame error: {e}")
                time.sleep(0.1)
    except GeneratorExit:
        pass
    finally:
        current_result = None
        release_camera()


@app.route("/")
def index():
    return send_from_directory(static_folder, "index.html")


@app.route("/assets/<path:path>")
def assets(path):
    response = send_from_directory(static_folder / "assets", path)
    response.headers["Cache-Control"] = "public, max-age=31536000"
    return response


@app.route("/manifest.json")
def manifest():
    response = send_from_directory(static_folder.parent / "frontend/public", "manifest.json")
    response.headers["Content-Type"] = "application/manifest+json"
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/vite.svg")
def vite_svg():
    return send_from_directory(static_folder, "vite.svg")


@app.route("/api/status")
def get_status():
    return jsonify({"running": is_streaming, "has_model": emotion_model is not None})


@app.route("/api/start", methods=["POST"])
def start_stream():
    global is_streaming, current_result
    try:
        current_result = None
        is_streaming = True
        cam = get_camera()
        if cam is None or not cam.isOpened():
            raise RuntimeError("Cannot open camera. Check OS permissions.")
        return jsonify({"status": "started", "has_model": emotion_model is not None})
    except Exception as e:
        is_streaming = False
        release_camera()
        return jsonify({"error": str(e), "has_model": emotion_model is not None}), 500


@app.route("/api/stop", methods=["POST"])
def stop_stream():
    global is_streaming, current_result
    is_streaming = False
    current_result = None
    release_camera()
    return jsonify({"status": "stopped"})


@app.route("/api/emotions")
def get_emotions():
    """Return current emotion detection results."""
    if not is_streaming or current_result is None:
        return jsonify({"results": []})

    return jsonify(
        {
            "results": [
                {
                    **current_result,
                    "face": {"x": 100, "y": 100, "width": 200, "height": 200, "confidence": 0.95},
                }
            ]
        }
    )


@app.route("/video_feed")
def video_feed():
    """Stream video from camera."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store"},
    )


@app.errorhandler(404)
def not_found(e):
    path = request.path
    if path.startswith(("/api/", "/assets/")) or path in (
        "/video_feed",
        "/manifest.json",
        "/vite.svg",
    ):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(static_folder, "index.html")


if __name__ == "__main__":
    print("=" * 60)
    print("🎭 Emotion AI - Real Detection Server")
    print("=" * 60)
    print("✅ Frontend: http://localhost:8080")
    print(f"✅ TensorFlow: {'Yes' if HAS_TF else 'No'}")
    print(f"✅ AI Model: {'Loaded' if emotion_model else 'Not found (using random)'}")
    print(f"✅ Face Detection: {'Ready' if face_cascade else 'Not available'}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    try:
        app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
    finally:
        release_camera()
