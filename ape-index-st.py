import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import math
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(layout="wide")
st.title("🦍 Ape Index Calculator")

# -------------------------
# Constants
# -------------------------
APEX_MEAN = 1.0
APEX_SD = 0.05
BUFFER_SIZE = 20
FREEZE_DURATION = 1.5  # seconds to freeze frame

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# -------------------------
# Helper functions
# -------------------------
def to_pixel(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height])

def distance(a, b):
    return np.linalg.norm(a - b)

def is_t_pose(lm_px, tol_px=80):
    l_sh = lm_px[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm_px[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_wr = lm_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wr = lm_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    return abs(l_sh[1] - l_wr[1]) < tol_px and abs(r_sh[1] - r_wr[1]) < tol_px

def draw_bell_curve(frame, ape_index):
    h, w, _ = frame.shape
    curve_w, curve_h = 200, 100
    start_x, start_y = w - curve_w - 20, h - curve_h - 20
    x_vals = np.linspace(APEX_MEAN - 3*APEX_SD, APEX_MEAN + 3*APEX_SD, curve_w)
    pdf = 1/(APEX_SD*np.sqrt(2*np.pi)) * np.exp(-0.5*((x_vals-APEX_MEAN)/APEX_SD)**2)
    pdf_scaled = (pdf / pdf.max() * curve_h).astype(int)
    pts = [(start_x+i, start_y+curve_h - pdf_scaled[i]) for i in range(curve_w)]
    for i in range(len(pts)-1):
        cv2.line(frame, pts[i], pts[i+1], (180,180,255), 2)
    rel_pos = (ape_index - (APEX_MEAN-3*APEX_SD)) / (6*APEX_SD)
    rel_pos = max(0.0, min(1.0, rel_pos))
    ape_x = int(rel_pos*curve_w) + start_x
    cv2.line(frame, (ape_x, start_y), (ape_x, start_y+curve_h), (0,0,255), 2)
    percentile = int((0.5*(1+math.erf((ape_index-APEX_MEAN)/(APEX_SD*math.sqrt(2)))))*100)
    cv2.putText(frame, f"{percentile}th percentile", (start_x, start_y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# -------------------------
# Video Processor
# -------------------------
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)
        self.ape_buffer = deque(maxlen=BUFFER_SIZE)
        self.arm_buffer = deque(maxlen=BUFFER_SIZE)
        self.height_buffer = deque(maxlen=BUFFER_SIZE)
        self.frozen_frame = None
        self.pose_start_time = None
        self.is_frozen = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)

        t_pose_detected = False
        ape_index_avg = None

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            h, w, _ = img.shape
            lm_px = {i: to_pixel(lm[i], w, h) for i in range(len(lm))}
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if is_t_pose(lm_px):
                t_pose_detected = True
                # Arm span
                l_wrist = lm_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
                r_wrist = lm_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                l_index = lm_px[mp_pose.PoseLandmark.LEFT_INDEX.value]
                r_index = lm_px[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                left_x = min(l_wrist[0], l_index[0])
                right_x = max(r_wrist[0], r_index[0])
                arm_span_px = distance(np.array([left_x,l_wrist[1]]), np.array([right_x,r_wrist[1]]))
                # Height
                head_y = min(lm_px[mp_pose.PoseLandmark.NOSE.value][1],
                             lm_px[mp_pose.PoseLandmark.LEFT_EAR.value][1],
                             lm_px[mp_pose.PoseLandmark.RIGHT_EAR.value][1])
                l_ankle = lm_px[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                r_ankle = lm_px[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                ankle_mid = (l_ankle + r_ankle)/2
                height_px = ankle_mid[1] - head_y

                if height_px > 0:
                    ape_index = arm_span_px / height_px
                    self.ape_buffer.append(ape_index)
                    self.arm_buffer.append(arm_span_px)
                    self.height_buffer.append(height_px)
                    ape_index_avg = np.mean(self.ape_buffer)

                    # Draw measurement lines
                    cv2.line(img, (int((left_x+right_x)/2), int(head_y)),
                             tuple(ankle_mid.astype(int)), (0,0,255), 3)
                    cv2.line(img, (int(left_x), int(l_wrist[1])),
                             (int(right_x), int(r_wrist[1])), (255,0,0), 3)

        # Freeze logic
        if t_pose_detected:
            if self.pose_start_time is None:
                self.pose_start_time = time.time()
            elif (time.time() - self.pose_start_time) >= FREEZE_DURATION:
                if ape_index_avg is not None:
                    self.frozen_frame = img.copy()
                self.is_frozen = True
        else:
            self.pose_start_time = None
            self.is_frozen = False

        final_frame = self.frozen_frame if self.is_frozen and self.frozen_frame is not None else img

        # Draw bell curve & text
        if ape_index_avg is not None:
            draw_bell_curve(final_frame, ape_index_avg)
            # Display result card
            percentile = int((0.5*(1+math.erf((ape_index_avg-APEX_MEAN)/(APEX_SD*math.sqrt(2)))))*100)
            st.session_state["ape_index"] = ape_index_avg
            st.session_state["percentile"] = percentile

        return av.VideoFrame.from_ndarray(final_frame, format="bgr24")

# -------------------------
# UI Elements
# -------------------------
if "ape_index" not in st.session_state:
    st.session_state["ape_index"] = None
if "percentile" not in st.session_state:
    st.session_state["percentile"] = None

cols = st.columns([3,1])
with cols[1]:
    if st.button("Mirror Webcam"):
        if "mirror" not in st.session_state:
            st.session_state["mirror"] = True
        else:
            st.session_state["mirror"] = not st.session_state["mirror"]

# Display results
if st.session_state["ape_index"] is not None:
    st.markdown(
        f"### 🦍 Your Ape Index: {st.session_state['ape_index']:.2f}\n"
        f"**Percentile:** {st.session_state['percentile']}th"
    )

# -------------------------
# Start WebRTC
# -------------------------
webrtc_streamer(
    key="ape_index_stream",
    video_processor_factory=PoseProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
        },
        "audio": False,
    },
)
