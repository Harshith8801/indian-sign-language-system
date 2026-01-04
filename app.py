import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import joblib
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import deque
from statistics import mode

# ------------------ CONFIG ------------------
LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'Hello','Thank-you','Yes','No'
]

# ------------------ LOAD MODEL (ONCE) ------------------
@st.cache_resource
def load_model():
    clf = joblib.load("model/gesture_classifier.pkl")
    scaler = joblib.load("model/gesture_scaler.pkl")
    return clf, scaler

clf, scaler = load_model()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ------------------ VIDEO PROCESSOR ------------------
class HandGestureDetector(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.prediction_buffer = deque(maxlen=7)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        blurred = cv2.GaussianBlur(img, (55, 55), 0)
        final_output = blurred.copy()
        stable_prediction = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

            xs, ys = [], []
            for p in lm.landmark:
                xs.append(int(p.x * w))
                ys.append(int(p.y * h))

            min_x, max_x = max(min(xs)-30, 0), min(max(xs)+30, w)
            min_y, max_y = max(min(ys)-30, 0), min(max(ys)+30, h)

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[min_y:max_y, min_x:max_x] = 255
            final_output = np.where(mask[..., None] == 255, img, blurred)

            features = []
            for p in lm.landmark:
                features.extend([p.x, p.y, p.z])

            if len(features) == 63:
                X = np.array(features).reshape(1, -1)
                X_scaled = scaler.transform(X)
                pred_idx = int(clf.predict(X_scaled)[0])
                pred = LABELS[pred_idx]

                self.prediction_buffer.append(pred)
                if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                    stable_prediction = mode(self.prediction_buffer)

        if stable_prediction:
            cv2.putText(
                final_output,
                f"Prediction: {stable_prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (0, 255, 0), 3
            )

        return final_output

# ------------------ STREAMLIT UI ------------------
st.title("🤟 ISL Hand Gesture Detection")

webrtc_streamer(
    key="gesture",
    video_processor_factory=HandGestureDetector,
    media_stream_constraints={"video": True, "audio": False},
)
