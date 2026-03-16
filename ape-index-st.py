import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import math
from collections import deque

st.title("Ape Index Calculator")

# -------------------------
# Constants
# -------------------------
APEX_MEAN = 1.0
APEX_SD = 0.05
BUFFER_SIZE = 20

ape_buffer = deque(maxlen=BUFFER_SIZE)
arm_buffer = deque(maxlen=BUFFER_SIZE)
height_buffer = deque(maxlen=BUFFER_SIZE)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def to_pixel(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height])


def distance(a, b):
    return np.linalg.norm(a - b)


def draw_bell_curve(frame, ape_index):
    h, w, _ = frame.shape
    curve_w, curve_h = 200, 100
    start_x, start_y = w - curve_w - 20, h - curve_h - 20

    x_vals = np.linspace(APEX_MEAN - 3 * APEX_SD, APEX_MEAN + 3 * APEX_SD, curve_w)
    pdf = 1/(APEX_SD*np.sqrt(2*np.pi)) * np.exp(-0.5*((x_vals-APEX_MEAN)/APEX_SD)**2)
    pdf_scaled = (pdf / pdf.max() * curve_h).astype(int)

    pts = []
    for i in range(curve_w):
        px = start_x + i
        py = start_y + curve_h - pdf_scaled[i]
        pts.append((px, py))

    for i in range(len(pts)-1):
        cv2.line(frame, pts[i], pts[i+1], (180,180,255), 2)

    rel_pos = (ape_index - (APEX_MEAN - 3*APEX_SD)) / (6*APEX_SD)
    rel_pos = max(0.0, min(1.0, rel_pos))
    ape_x = int(rel_pos * curve_w) + start_x

    cv2.line(frame, (ape_x, start_y), (ape_x, start_y+curve_h), (0,0,255), 2)


class PoseProcessor(VideoProcessorBase):

    def __init__(self):
        self.pose = mp_pose.Pose()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.pose.process(frame_rgb)

        if result.pose_landmarks:

            lm = result.pose_landmarks.landmark
            h, w, _ = img.shape

            lm_px = {i: to_pixel(lm[i], w, h) for i in range(len(lm))}

            mp_drawing.draw_landmarks(
                img,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            l_wrist = lm_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
            r_wrist = lm_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            l_index = lm_px[mp_pose.PoseLandmark.LEFT_INDEX.value]
            r_index = lm_px[mp_pose.PoseLandmark.RIGHT_INDEX.value]

            left_x = min(l_wrist[0], l_index[0])
            right_x = max(r_wrist[0], r_index[0])

            arm_span_px = distance(
                np.array([left_x, l_wrist[1]]),
                np.array([right_x, r_wrist[1]])
            )

            head_y = min(
                lm_px[mp_pose.PoseLandmark.NOSE.value][1],
                lm_px[mp_pose.PoseLandmark.LEFT_EAR.value][1],
                lm_px[mp_pose.PoseLandmark.RIGHT_EAR.value][1]
            )

            l_ankle = lm_px[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            r_ankle = lm_px[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            ankle_mid = (l_ankle + r_ankle) / 2
            height_px = ankle_mid[1] - head_y

            if height_px > 0:

                ape_index = arm_span_px / height_px

                ape_buffer.append(ape_index)
                arm_buffer.append(arm_span_px)
                height_buffer.append(height_px)

                ape_index_avg = np.mean(ape_buffer)

                cv2.putText(
                    img,
                    f"Ape Index: {ape_index_avg:.2f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                draw_bell_curve(img, ape_index_avg)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="pose",
    video_processor_factory=PoseProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
