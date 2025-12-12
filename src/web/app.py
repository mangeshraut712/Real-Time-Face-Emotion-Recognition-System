#!/usr/bin/env python3
"""
Flask Web Application for Emotion Recognition.

This module provides a modern web-based interface for real-time
facial emotion detection with video streaming.

Features:
- SPA (Single Page Application) serving
- Real-time video streaming (MJPEG)
- REST API definition
- CORS support
- Static asset serving with cache control
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from src.core.emotion_detector import EmotionDetector
from src.core.face_detector import FaceDetector

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np

logger = logging.getLogger(__name__)


class VideoStream:
    """Thread-safe video stream handler."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.results: list = []
        self.running = False
        self.lock = threading.Lock()

        # Detectors
        self.face_detector: FaceDetector | None = None
        self.emotion_detector: EmotionDetector | None = None

    def start(self):
        """Start the video stream."""
        if self.running:
            return

        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        # Initialize detectors
        self.face_detector = FaceDetector()
        self.emotion_detector = EmotionDetector()

        self.running = True

        # Start capture thread
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

        logger.info("Video stream started")

    def _capture_loop(self):
        """Capture and process frames."""
        while self.running:
            if self.capture is None:
                break

            ret, frame = self.capture.read()
            if not ret:
                continue

            # Resize for performance
            frame = cv2.resize(frame, (640, 480))

            # Detect
            faces = self.face_detector.detect(frame)
            results = []

            if faces:
                results = self.emotion_detector.detect_emotions(frame, faces)

                # Draw results on the frame for the video feed
                # Note: The frontend also receives raw data, but this provides a visual debugging feed
                for result in results:
                    face = result.face
                    color = result.color

                    # Modern rounded rectangle for face
                    # (Simple cv2 doesn't support rounded, but we can keep it clean)
                    cv2.rectangle(
                        frame,
                        (face.x, face.y),
                        (face.x + face.width, face.y + face.height),
                        color,
                        2,
                    )

            with self.lock:
                self.frame = frame
                self.results = results

            time.sleep(0.01)  # ~100 FPS max limit

    def get_frame(self) -> bytes | None:
        """Get the current frame as JPEG bytes."""
        with self.lock:
            if self.frame is None:
                return None

            # High quality JPEG for better visuals
            ret, jpeg = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                return None

            return jpeg.tobytes()

    def get_results(self) -> list:
        """Get the current emotion results."""
        with self.lock:
            return [r.as_dict() for r in self.results]

    def stop(self):
        """Stop the video stream."""
        self.running = False

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        # Clear buffers so UI doesn't show stale data
        with self.lock:
            self.frame = None
            self.results = []

        # Release detector resources
        if self.face_detector is not None:
            with contextlib.suppress(Exception):
                self.face_detector.close()
            self.face_detector = None

        if self.emotion_detector is not None:
            self.emotion_detector = None

        logger.info("Video stream stopped")


# Global video stream instance
video_stream: VideoStream | None = None


def create_app() -> Flask:
    """Create and configure the Flask application."""
    # Define paths based on new structure
    static_folder = Path(__file__).parent / "static"
    template_folder = static_folder

    app = Flask(
        __name__,
        static_url_path="",
        static_folder=str(static_folder),
        template_folder=str(template_folder),
    )

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------

    @app.route("/")
    def index():
        """Serve the React App."""
        assets_dir = static_folder / "assets"
        has_assets = assets_dir.exists() and any(p.suffix == ".js" for p in assets_dir.iterdir())
        if not has_assets:
            return (
                "<h1>Frontend not built</h1>"
                "<p>Run: <code>cd src/web/frontend && npm install && npm run build</code></p>",
                500,
            )
        return render_template("index.html")

    @app.route("/assets/<path:path>")
    def serve_assets(path):
        """
        Serve static assets with long cache headers.
        Vite hashes filenames, so we can cache aggressively.
        """
        response = send_from_directory(str(static_folder / "assets"), path)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @app.route("/manifest.json")
    def manifest():
        """Serve PWA manifest if available."""
        public_manifest = Path(__file__).parent / "frontend" / "public" / "manifest.json"
        if public_manifest.exists():
            response = send_from_directory(str(public_manifest.parent), public_manifest.name)
            response.headers["Content-Type"] = "application/manifest+json"
            response.headers["Cache-Control"] = "no-cache"
            return response
        # Fallback: if manifest is in static root
        if (static_folder / "manifest.json").exists():
            response = send_from_directory(str(static_folder), "manifest.json")
            response.headers["Content-Type"] = "application/manifest+json"
            response.headers["Cache-Control"] = "no-cache"
            return response
        return jsonify({"error": "manifest not found"}), 404

    @app.route("/vite.svg")
    def vite_svg():
        """Serve favicon/svg from static or public."""
        public_icon = Path(__file__).parent / "frontend" / "public" / "vite.svg"
        if public_icon.exists():
            return send_from_directory(str(public_icon.parent), public_icon.name)
        return send_from_directory(str(static_folder), "vite.svg")

    @app.errorhandler(404)
    def not_found(e):
        """
        Catch-all route for SPA client-side routing.
        If the path isn't an API or asset, serve index.html.
        """
        path = request.path
        if path.startswith(("/api/", "/assets/")) or path in (
            "/video_feed",
            "/manifest.json",
            "/vite.svg",
        ):
            return jsonify({"error": "not found"}), 404
        return render_template("index.html")

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------

    @app.route("/video_feed")
    def video_feed():
        """Stream video frames via MJPEG."""

        def generate() -> Generator[bytes, None, None]:
            while True:
                if video_stream is None:
                    time.sleep(0.1)
                    continue

                frame = video_stream.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

                time.sleep(0.033)  # Limit to ~30 FPS for browser performance

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/emotions")
    def get_emotions():
        """Get current emotion detection results JSON."""
        if video_stream is None:
            # Return empty if not running, but with valid structure
            return jsonify({"results": []})

        results = video_stream.get_results()
        return jsonify({"results": results})

    @app.route("/api/start", methods=["POST"])
    def start_stream():
        """Start the video stream."""
        global video_stream

        camera_index = request.json.get("camera_index", 0) if request.json else 0

        try:
            if video_stream is not None:
                video_stream.stop()

            video_stream = VideoStream(camera_index)
            video_stream.start()

            return jsonify({"status": "started"})
        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/stop", methods=["POST"])
    def stop_stream():
        """Stop the video stream."""
        global video_stream

        if video_stream is not None:
            video_stream.stop()
            video_stream = None

        return jsonify({"status": "stopped"})

    @app.route("/api/status")
    def get_status():
        """Get stream status."""
        if video_stream is None:
            return jsonify({"running": False})

        return jsonify({"running": video_stream.running})

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
    auto_start: bool = False,
):
    """
    Run the Flask development server.
    """
    global video_stream

    # Configure production-ready logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    app = create_app()

    # Auto-start video stream if requested
    if auto_start:
        try:
            video_stream = VideoStream()
            video_stream.start()
        except Exception as e:
            logger.warning(f"Could not auto-start camera: {e}")

    logger.info(f"🚀 Server running at http://{host}:{port}")
    logger.info("hit CTRL+C to stop")

    try:
        app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
    finally:
        if video_stream is not None:
            video_stream.stop()


if __name__ == "__main__":
    run_server(debug=True)
