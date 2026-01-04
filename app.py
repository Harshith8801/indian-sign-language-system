import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import requests
from collections import deque
from statistics import mode

BACKEND_URL = "http://127.0.0.1:8000/predict"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class HandGestureDetector(VideoTransformerBase):

    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.prediction_buffer = deque(maxlen=7)

    def send_to_backend(self, features):
        try:
            response = requests.post(
                BACKEND_URL,
                json={"features": features}
            )
            return response.json().get("prediction", None)
        except:
            return None  # Backend off

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

            mask = np.zeros((h, w), dtype=np.uint8)
            xs, ys = [], []

            for p in lm.landmark:
                xs.append(int(p.x * w))
                ys.append(int(p.y * h))

            min_x, max_x = max(min(xs)-30, 0), min(max(xs)+30, w)
            min_y, max_y = max(min(ys)-30, 0), min(max(ys)+30, h)

            mask[min_y:max_y, min_x:max_x] = 255
            final_output = np.where(mask[..., None] == 255, img, blurred)

            # Extract 63 features
            features = []
            for p in lm.landmark:
                features.extend([p.x, p.y, p.z])

            if len(features) == 63:
                pred = self.send_to_backend(features)

                if pred is not None:
                    self.prediction_buffer.append(pred)

                    if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                        stable_prediction = mode(self.prediction_buffer)

        if stable_prediction is not None:
            cv2.putText(
                final_output,
                f"Prediction: {stable_prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (0, 255, 0), 3
            )

        return final_output

# STREAMLIT UI
st.title("🤟 ISL Hand Gesture Number Detection (API Connected)")
st.info("Backend must be running!")

webrtc_streamer(
    key="gesture-detection",
    video_processor_factory=HandGestureDetector,
    media_stream_constraints={"video": True, "audio": False},
)
# To run this app, use the command:
# cd C:\ISL_Project
#streamlit run app.py
