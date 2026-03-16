import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import threading

# -------------------------
# Constants
# -------------------------
BUFFER_SIZE = 20
FREEZE_DURATION = 1.5
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# -------------------------
# Thread-safe flags
# -------------------------
class SharedFlags:
    def __init__(self):
        self.mirror = False
        self.unfreeze = False
        self.lock = threading.Lock()

shared_flags = SharedFlags()

# -------------------------
# Buttons
# -------------------------
cols = st.columns([3,1])
with cols[1]:
    if st.button("Mirror Webcam"):
        with shared_flags.lock:
            shared_flags.mirror = not shared_flags.mirror
    if st.button("Unfreeze"):
        with shared_flags.lock:
            shared_flags.unfreeze = True

st.text(f"Mirror: {shared_flags.mirror}, Unfreeze: {shared_flags.unfreeze}")

# -------------------------
# Helpers
# -------------------------
def to_pixel(landmark, width, height):
    return np.array([landmark.x*width, landmark.y*height])

def distance(a,b):
    return np.linalg.norm(a-b)

def is_t_pose(lm_px, img_height, tol_ratio=0.1):
    tol_px = img_height * tol_ratio
    l_sh = lm_px[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm_px[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_wr = lm_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wr = lm_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    return abs(l_sh[1]-l_wr[1])<tol_px and abs(r_sh[1]-r_wr[1])<tol_px

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
        self.is_frozen = False
        self.pose_start_time = None

    def recv(self, frame):
        # Thread-safe flags
        with shared_flags.lock:
            mirror_flag = shared_flags.mirror
            unfreeze_flag = shared_flags.unfreeze
            if unfreeze_flag:
                shared_flags.unfreeze = False

        # Unfreeze
        if unfreeze_flag:
            self.is_frozen = False
            self.frozen_frame = None
            print("Unfreeze triggered!")

        # Convert frame and resize for display
        img = frame.to_ndarray(format="bgr24")
        if mirror_flag:
            img = cv2.flip(img, 1)

        display_frame = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        scale_x = DISPLAY_WIDTH / img.shape[1]
        scale_y = DISPLAY_HEIGHT / img.shape[0]

        # MediaPipe
        result = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t_pose_detected = False
        ape_index_avg = None

        if result.pose_landmarks:
            # Draw landmarks scaled to display frame
            for lm in result.pose_landmarks.landmark:
                x = int(lm.x * DISPLAY_WIDTH)
                y = int(lm.y * DISPLAY_HEIGHT)
                cv2.circle(display_frame, (x, y), 5, (0,255,0), -1)

            # Draw connections
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                x1, y1 = int(result.pose_landmarks.landmark[start_idx].x * DISPLAY_WIDTH), int(result.pose_landmarks.landmark[start_idx].y * DISPLAY_HEIGHT)
                x2, y2 = int(result.pose_landmarks.landmark[end_idx].x * DISPLAY_WIDTH), int(result.pose_landmarks.landmark[end_idx].y * DISPLAY_HEIGHT)
                cv2.line(display_frame, (x1, y1), (x2, y2), (0,0,255), 2)

            # T-pose detection
            lm_px = {i: to_pixel(lm, DISPLAY_WIDTH, DISPLAY_HEIGHT) for i, lm in enumerate(result.pose_landmarks.landmark)}
            if is_t_pose(lm_px, DISPLAY_HEIGHT):
                t_pose_detected = True
                l_wrist = lm_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
                r_wrist = lm_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                arm_span_px = distance(l_wrist,r_wrist)
                head_y = lm_px[mp_pose.PoseLandmark.NOSE.value][1]
                l_ankle = lm_px[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                r_ankle = lm_px[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                height_px = (l_ankle+r_ankle)[1]/2 - head_y

                if height_px>0:
                    ape_index = arm_span_px/height_px
                    self.ape_buffer.append(ape_index)
                    self.arm_buffer.append(arm_span_px)
                    self.height_buffer.append(height_px)
                    ape_index_avg = np.mean(self.ape_buffer)
                    print(f"T-pose detected, ape index avg: {ape_index_avg:.2f}")

        # Freeze logic
        if t_pose_detected:
            if self.pose_start_time is None:
                self.pose_start_time = time.time()
            elif (time.time()-self.pose_start_time) >= FREEZE_DURATION:
                if ape_index_avg is not None:
                    self.frozen_frame = display_frame.copy()
                self.is_frozen = True
        else:
            self.pose_start_time = None

        if self.is_frozen and self.frozen_frame is not None:
            display_frame = self.frozen_frame.copy()

        display_frame = np.ascontiguousarray(display_frame, dtype=np.uint8)

        # Overlay text
        if self.is_frozen:
            cv2.putText(display_frame, "*Frozen*", (DISPLAY_WIDTH-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if mirror_flag:
            cv2.putText(display_frame, "Mirror ON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        print("Frame processed:", display_frame.shape)
        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")

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
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
)
