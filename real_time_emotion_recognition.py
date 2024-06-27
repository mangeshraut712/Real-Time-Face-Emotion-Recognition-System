from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='cv2')

# Paths to the face detection model and the emotion classification model
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# Load the face detection model
face_detection = cv2.CascadeClassifier(detection_model_path)
# Load the emotion classification model
emotion_classifier = load_model(emotion_model_path, compile=False)

# List of emotions
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Emotion Detection")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.video_capture = cv2.VideoCapture(0)

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame, width=640, height=480)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.label = ttk.Label(self.main_frame, text="Emotion Probabilities:", font=("Helvetica", 16))
        self.label.pack(side=tk.TOP, pady=10)

        self.prob_canvas = tk.Canvas(self.main_frame, width=640, height=250)
        self.prob_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.button_frame = ttk.Frame(self.main_frame, padding="10")
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.on_closing)
        self.stop_button.pack(side=tk.RIGHT, padx=10)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        frameClone = frame.copy()

        if len(faces) > 0:
            (fX, fY, fW, fH) = sorted(faces, reverse=True, key=lambda x: (x[2] * x[3]))[0]
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            self.prob_canvas.delete("all")  # Clear previous content
            for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 600)  # Adjusted width for better visibility
                self.prob_canvas.create_rectangle(50, (i * 35) + 5, 50 + w, (i * 35) + 35, fill="red")
                self.prob_canvas.create_text(55, (i * 35) + 20, anchor="w", text=text, fill="white", font=("Helvetica", 14))
            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        frame = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
        self.canvas.image = frame

        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.video_capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
