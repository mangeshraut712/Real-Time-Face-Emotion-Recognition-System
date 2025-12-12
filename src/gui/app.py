"""
Modern GUI Application for Emotion Recognition.

This module provides a beautiful, dark-themed GUI application
for real-time facial emotion detection.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from src.core.emotion_detector import EMOTIONS, EmotionDetector, EmotionResult
from src.core.face_detector import FaceDetector
from src.utils.video import capture_screenshot

logger = logging.getLogger(__name__)

# Theme colors
THEME = {
    "bg_primary": "#0d1117",
    "bg_secondary": "#161b22",
    "bg_tertiary": "#21262d",
    "text_primary": "#f0f6fc",
    "text_secondary": "#8b949e",
    "accent": "#58a6ff",
    "border": "#30363d",
    "success": "#2ea043",
    "warning": "#d29922",
    "error": "#f85149",
}

# Emotion colors in hex for GUI
EMOTION_HEX_COLORS = {
    "angry": "#e74c3c",
    "disgust": "#9b59b6",
    "scared": "#f1c40f",
    "happy": "#2ecc71",
    "sad": "#3498db",
    "surprised": "#e67e22",
    "neutral": "#95a5a6",
}

# Emotion emojis
EMOTION_EMOJIS = {
    "angry": "😠",
    "disgust": "🤢",
    "scared": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprised": "😲",
    "neutral": "😐",
}


class ModernButton(tk.Canvas):
    """Custom modern-styled button."""

    def __init__(
        self,
        parent,
        text: str,
        command=None,
        width: int = 120,
        height: int = 36,
        bg: str = THEME["accent"],
        fg: str = THEME["text_primary"],
        hover_bg: str = "#4c94e6",
        **kwargs,
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=THEME["bg_primary"],
            highlightthickness=0,
            **kwargs,
        )

        self.command = command
        self.bg = bg
        self.fg = fg
        self.hover_bg = hover_bg
        self.text = text
        self.width = width
        self.height = height

        self._draw_button(bg)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _draw_button(self, color: str):
        self.delete("all")

        # Draw rounded rectangle
        radius = 8
        self.create_arc(
            0,
            0,
            radius * 2,
            radius * 2,
            start=90,
            extent=90,
            fill=color,
            outline=color,
        )
        self.create_arc(
            self.width - radius * 2,
            0,
            self.width,
            radius * 2,
            start=0,
            extent=90,
            fill=color,
            outline=color,
        )
        self.create_arc(
            0,
            self.height - radius * 2,
            radius * 2,
            self.height,
            start=180,
            extent=90,
            fill=color,
            outline=color,
        )
        self.create_arc(
            self.width - radius * 2,
            self.height - radius * 2,
            self.width,
            self.height,
            start=270,
            extent=90,
            fill=color,
            outline=color,
        )

        # Fill rectangles
        self.create_rectangle(
            radius,
            0,
            self.width - radius,
            self.height,
            fill=color,
            outline=color,
        )
        self.create_rectangle(
            0,
            radius,
            self.width,
            self.height - radius,
            fill=color,
            outline=color,
        )

        # Draw text
        self.create_text(
            self.width // 2,
            self.height // 2,
            text=self.text,
            fill=self.fg,
            font=("Helvetica", 11, "bold"),
        )

    def _on_enter(self, event):
        self._draw_button(self.hover_bg)

    def _on_leave(self, event):
        self._draw_button(self.bg)

    def _on_click(self, event):
        if self.command:
            self.command()


class EmotionBarChart(tk.Canvas):
    """Animated horizontal bar chart for emotion probabilities."""

    def __init__(
        self,
        parent,
        width: int = 400,
        height: int = 280,
        **kwargs,
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=THEME["bg_secondary"],
            highlightthickness=0,
            **kwargs,
        )

        self.width = width
        self.height = height
        self.bar_height = 28
        self.bar_spacing = 12
        self.label_width = 90
        self.margin = 15

        self._current_values: dict[str, float] = dict.fromkeys(EMOTIONS, 0.0)
        self._target_values: dict[str, float] = dict.fromkeys(EMOTIONS, 0.0)
        self._animating = False

        self._draw_empty()

    def _draw_empty(self):
        """Draw empty chart with labels."""
        self.delete("all")

        for i, emotion in enumerate(EMOTIONS):
            y = self.margin + i * (self.bar_height + self.bar_spacing)

            # Emoji
            self.create_text(
                20,
                y + self.bar_height // 2,
                text=EMOTION_EMOJIS[emotion],
                font=("Helvetica", 14),
                anchor="w",
            )

            # Label
            self.create_text(
                45,
                y + self.bar_height // 2,
                text=emotion.capitalize(),
                fill=THEME["text_primary"],
                font=("Helvetica", 11),
                anchor="w",
            )

            # Background bar
            bar_x = self.label_width
            bar_width = self.width - self.label_width - self.margin - 50

            self.create_rectangle(
                bar_x,
                y + 2,
                bar_x + bar_width,
                y + self.bar_height - 2,
                fill=THEME["bg_tertiary"],
                outline="",
                tags=f"bg_{emotion}",
            )

    def update_values(self, probabilities: dict[str, float]):
        """Update chart with new probability values."""
        self._target_values = probabilities.copy()

        if not self._animating:
            self._animate()

    def _animate(self):
        """Animate bar values to targets."""
        self._animating = True

        # Smoothly interpolate current values to targets
        needs_update = False
        smooth_factor = 0.15

        for emotion in EMOTIONS:
            diff = self._target_values[emotion] - self._current_values[emotion]
            if abs(diff) > 0.001:
                self._current_values[emotion] += diff * smooth_factor
                needs_update = True

        self._draw_bars()

        if needs_update:
            self.after(16, self._animate)  # ~60 FPS
        else:
            self._animating = False

    def _draw_bars(self):
        """Draw the current bar values."""
        self.delete("bars")
        self.delete("values")

        bar_x = self.label_width
        max_bar_width = self.width - self.label_width - self.margin - 50

        for i, emotion in enumerate(EMOTIONS):
            y = self.margin + i * (self.bar_height + self.bar_spacing)
            value = self._current_values[emotion]
            bar_width = max(0, value * max_bar_width)

            if bar_width > 0:
                # Draw colored bar
                color = EMOTION_HEX_COLORS[emotion]
                self.create_rectangle(
                    bar_x,
                    y + 2,
                    bar_x + bar_width,
                    y + self.bar_height - 2,
                    fill=color,
                    outline="",
                    tags="bars",
                )

            # Draw percentage
            self.create_text(
                self.width - self.margin,
                y + self.bar_height // 2,
                text=f"{value * 100:.1f}%",
                fill=THEME["text_primary"],
                font=("Helvetica", 10, "bold"),
                anchor="e",
                tags="values",
            )


class EmotionApp:
    """
    Modern GUI application for real-time emotion detection.

    Features:
    - Dark mode interface
    - Animated probability chart
    - Multi-face support
    - Screenshot capture
    - Emotion history
    """

    def __init__(
        self,
        camera_index: int = 0,
        detection_model_path: Path | None = None,
        emotion_model_path: Path | None = None,
    ):
        """
        Initialize the application.

        Args:
            camera_index: Camera index for OpenCV
            detection_model_path: Path to face detection model
            emotion_model_path: Path to emotion model
        """
        self.camera_index = camera_index
        self.running = False

        # Initialize detectors
        try:
            self.face_detector = FaceDetector(
                cascade_path=detection_model_path,
            )
            self.emotion_detector = EmotionDetector(
                model_path=emotion_model_path,
            )
        except FileNotFoundError as e:
            messagebox.showerror("Model Error", str(e))
            raise

        # Video capture
        self.video_capture: cv2.VideoCapture | None = None

        # Emotion history
        self.emotion_history: list[str] = []
        self.max_history = 30

        # Create GUI
        self._create_gui()

    def _create_gui(self):
        """Create the main GUI."""
        self.root = tk.Tk()
        self.root.title("🎭 Real-Time Emotion Detection")
        self.root.configure(bg=THEME["bg_primary"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Set minimum size
        self.root.minsize(1100, 700)

        # Configure grid
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Left panel - Video feed
        self._create_video_panel()

        # Right panel - Info and controls
        self._create_info_panel()

        # Status bar
        self._create_status_bar()

    def _create_video_panel(self):
        """Create the video feed panel."""
        self.video_frame = tk.Frame(
            self.root,
            bg=THEME["bg_primary"],
            padx=15,
            pady=15,
        )
        self.video_frame.grid(row=0, column=0, sticky="nsew")

        # Title
        title_frame = tk.Frame(self.video_frame, bg=THEME["bg_primary"])
        title_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            title_frame,
            text="📹 Live Camera Feed",
            font=("Helvetica", 16, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_primary"],
        ).pack(side="left")

        # Video canvas
        self.canvas = tk.Canvas(
            self.video_frame,
            width=640,
            height=480,
            bg=THEME["bg_secondary"],
            highlightthickness=2,
            highlightbackground=THEME["border"],
        )
        self.canvas.pack(fill="both", expand=True)

        # Placeholder text
        self.canvas.create_text(
            320,
            240,
            text="Initializing camera...",
            fill=THEME["text_secondary"],
            font=("Helvetica", 14),
            tags="placeholder",
        )

    def _create_info_panel(self):
        """Create the information panel."""
        self.info_frame = tk.Frame(
            self.root,
            bg=THEME["bg_primary"],
            padx=15,
            pady=15,
        )
        self.info_frame.grid(row=0, column=1, sticky="nsew")

        # Current emotion display
        emotion_frame = tk.Frame(self.info_frame, bg=THEME["bg_secondary"], padx=15, pady=15)
        emotion_frame.pack(fill="x", pady=(0, 15))

        tk.Label(
            emotion_frame,
            text="Current Emotion",
            font=("Helvetica", 12),
            fg=THEME["text_secondary"],
            bg=THEME["bg_secondary"],
        ).pack()

        self.emoji_label = tk.Label(
            emotion_frame,
            text="😐",
            font=("Helvetica", 48),
            bg=THEME["bg_secondary"],
        )
        self.emoji_label.pack(pady=5)

        self.emotion_label = tk.Label(
            emotion_frame,
            text="Waiting...",
            font=("Helvetica", 18, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_secondary"],
        )
        self.emotion_label.pack()

        self.confidence_label = tk.Label(
            emotion_frame,
            text="",
            font=("Helvetica", 12),
            fg=THEME["text_secondary"],
            bg=THEME["bg_secondary"],
        )
        self.confidence_label.pack()

        # Probability chart
        chart_title = tk.Label(
            self.info_frame,
            text="📊 Emotion Probabilities",
            font=("Helvetica", 14, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_primary"],
        )
        chart_title.pack(fill="x", pady=(10, 5))

        self.prob_chart = EmotionBarChart(self.info_frame, width=350, height=290)
        self.prob_chart.pack(fill="x", pady=(0, 15))

        # History
        history_frame = tk.Frame(self.info_frame, bg=THEME["bg_secondary"], padx=10, pady=10)
        history_frame.pack(fill="x", pady=(0, 15))

        tk.Label(
            history_frame,
            text="📜 Recent History",
            font=("Helvetica", 12, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_secondary"],
        ).pack(anchor="w")

        self.history_canvas = tk.Canvas(
            history_frame,
            height=40,
            bg=THEME["bg_secondary"],
            highlightthickness=0,
        )
        self.history_canvas.pack(fill="x", pady=5)

        # Control buttons
        btn_frame = tk.Frame(self.info_frame, bg=THEME["bg_primary"])
        btn_frame.pack(fill="x", pady=10)

        self.screenshot_btn = ModernButton(
            btn_frame,
            text="📷 Screenshot",
            command=self.take_screenshot,
            width=140,
            bg=THEME["accent"],
        )
        self.screenshot_btn.pack(side="left", padx=5)

        self.stop_btn = ModernButton(
            btn_frame,
            text="⏹ Stop",
            command=self.on_closing,
            width=100,
            bg=THEME["error"],
            hover_bg="#d63031",
        )
        self.stop_btn.pack(side="right", padx=5)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_frame = tk.Frame(
            self.root,
            bg=THEME["bg_secondary"],
            height=30,
        )
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.status_label = tk.Label(
            self.status_frame,
            text="Ready",
            font=("Helvetica", 10),
            fg=THEME["text_secondary"],
            bg=THEME["bg_secondary"],
            padx=10,
        )
        self.status_label.pack(side="left")

        self.fps_label = tk.Label(
            self.status_frame,
            text="FPS: --",
            font=("Helvetica", 10),
            fg=THEME["text_secondary"],
            bg=THEME["bg_secondary"],
            padx=10,
        )
        self.fps_label.pack(side="right")

    def start(self):
        """Start the application."""
        # Open camera
        self.video_capture = cv2.VideoCapture(self.camera_index)

        if not self.video_capture.isOpened():
            messagebox.showerror(
                "Camera Error",
                f"Unable to open camera index {self.camera_index}.",
            )
            return

        self.running = True
        self.status_label.config(text="Camera active • Detecting emotions...")

        # Start frame updates
        self._frame_count = 0
        self._last_fps_time = cv2.getTickCount()
        self.update_frame()

        # Start main loop
        self.root.mainloop()

    def update_frame(self):
        """Update the video frame."""
        if not self.running or self.video_capture is None:
            return

        ret, frame = self.video_capture.read()

        if not ret:
            self.root.after(10, self.update_frame)
            return

        # Resize frame
        frame = cv2.resize(frame, (640, 480))

        # Detect faces
        faces = self.face_detector.detect(frame)

        # Process each face
        results = []
        if faces:
            results = self.emotion_detector.detect_emotions(frame, faces)

            # Draw results
            for result in results:
                face = result.face
                color = result.color

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (face.x, face.y),
                    (face.x + face.width, face.y + face.height),
                    color,
                    2,
                )

                # Draw label with background
                label = f"{result.emotion}: {result.confidence:.0%}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                cv2.rectangle(
                    frame,
                    (face.x, face.y - text_h - 10),
                    (face.x + text_w + 10, face.y),
                    color,
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (face.x + 5, face.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        # Update UI with first face result
        if results:
            result = results[0]
            self._update_emotion_display(result)

        # Convert frame for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk  # Keep reference

        # Update FPS
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            current_time = cv2.getTickCount()
            elapsed = (current_time - self._last_fps_time) / cv2.getTickFrequency()
            fps = 30 / elapsed
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self._last_fps_time = current_time

        # Schedule next update
        self.root.after(10, self.update_frame)

    def _update_emotion_display(self, result: EmotionResult):
        """Update the emotion display with new result."""
        # Update main emotion
        emoji = EMOTION_EMOJIS.get(result.emotion, "🤔")
        color = EMOTION_HEX_COLORS.get(result.emotion, THEME["text_primary"])

        self.emoji_label.config(text=emoji)
        self.emotion_label.config(text=result.emotion.upper(), fg=color)
        self.confidence_label.config(text=f"Confidence: {result.confidence:.1%}")

        # Update probability chart
        self.prob_chart.update_values(result.probabilities)

        # Update history
        self.emotion_history.append(result.emotion)
        if len(self.emotion_history) > self.max_history:
            self.emotion_history = self.emotion_history[-self.max_history :]

        self._draw_history()

    def _draw_history(self):
        """Draw the emotion history."""
        self.history_canvas.delete("all")

        width = self.history_canvas.winfo_width()
        if width <= 1:
            width = 330

        dot_size = 10
        spacing = (
            (width - 20) / max(1, len(self.emotion_history) - 1)
            if len(self.emotion_history) > 1
            else 20
        )

        for i, emotion in enumerate(self.emotion_history):
            x = 10 + i * min(spacing, 12)
            y = 20
            color = EMOTION_HEX_COLORS.get(emotion, THEME["text_secondary"])

            self.history_canvas.create_oval(
                x - dot_size // 2,
                y - dot_size // 2,
                x + dot_size // 2,
                y + dot_size // 2,
                fill=color,
                outline="",
            )

    def take_screenshot(self):
        """Take a screenshot of the current frame."""
        if self.video_capture is None:
            return

        ret, frame = self.video_capture.read()
        if ret:
            # Draw current detection
            faces = self.face_detector.detect(frame)
            if faces:
                results = self.emotion_detector.detect_emotions(frame, faces)
                frame = self.emotion_detector.draw_results(frame, results)

            # Save screenshot
            output_path = capture_screenshot(frame)
            self.status_label.config(text=f"Screenshot saved: {output_path.name}")

    def on_closing(self):
        """Handle window closing."""
        self.running = False

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        self.root.destroy()


def run_app(
    camera_index: int = 0,
    detection_model_path: Path | None = None,
    emotion_model_path: Path | None = None,
):
    """
    Run the emotion recognition application.

    Args:
        camera_index: Camera index for OpenCV
        detection_model_path: Path to face detection model
        emotion_model_path: Path to emotion model
    """
    try:
        app = EmotionApp(
            camera_index=camera_index,
            detection_model_path=detection_model_path,
            emotion_model_path=emotion_model_path,
        )
        app.start()
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_app()
