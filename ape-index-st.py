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

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# -------------------------
# Thread-safe shared flags
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

# Status text
st.text(f"Mirror: {shared_flags.mirror}, Unfreeze: {shared_flags.unfreeze}")

# -------------------------
# Helpers
# -------------------------
def to_pixel(landmark, width, height):
    return np.array([landmark.x*width, landmark.y*height])

def distance(a,b):
    return np.linalg.norm(a-b)

def is_t_pose(lm_px, img_height, tol_ratio=0.1):
    tol_px = img_height * tol_ratio  # tolerance relative to frame height
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
        # Thread-safe copy of flags
        with shared_flags.lock:
            mirror_flag = shared_flags.mirror
            unfreeze_flag = shared_flags.unfreeze
            if unfreeze_flag:
                shared_flags.unfreeze = False

        # Handle Unfreeze
        if unfreeze_flag:
            self.is_frozen = False
            self.frozen_frame = None
            print("Unfreeze triggered!")

        # Convert frame
        img = frame.to_ndarray(format="bgr24")
        if mirror_flag:
            img = cv2.flip(img, 1)

        # MediaPipe processing
        result = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t_pose_detected = False
        ape_index_avg = None

        # Copy frame for drawing
        final_frame = img.copy()

        if result.pose_landmarks:
            # Scale landmarks to pixel coordinates
            lm_px = {i: to_pixel(lm, final_frame.shape[1], final_frame.shape[0]) 
                     for i, lm in enumerate(result.pose_landmarks.landmark)}

            # Draw landmarks with dynamic thickness
            scale = final_frame.shape[0]/720
            mp_drawing.draw_landmarks(
                final_frame, 
                result.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=int(2*scale), circle_radius=int(2*scale)),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=int(2*scale))
            )

            # T-pose detection
            if is_t_pose(lm_px, final_frame.shape[0]):
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
                    self.frozen_frame = final_frame.copy()
                self.is_frozen = True
        else:
            self.pose_start_time = None

        # Show frozen frame if frozen
        if self.is_frozen and self.frozen_frame is not None:
            final_frame = self.frozen_frame.copy()

        # Ensure contiguous frame for WebRTC
        final_frame = np.ascontiguousarray(final_frame, dtype=np.uint8)

        # Overlay indicators
        scale = final_frame.shape[0]/720
        if self.is_frozen:
            cv2.putText(final_frame,"*Frozen*", (final_frame.shape[1]-180, int(40*scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1*scale,(0,0,255), int(2*scale))
        if mirror_flag:
            cv2.putText(final_frame,"Mirror ON", (10, int(40*scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1*scale,(0,255,0), int(2*scale))

        # Debug
        print("Frame received, shape:", final_frame.shape)

        return av.VideoFrame.from_ndarray(final_frame, format="bgr24")

# -------------------------
# Start WebRTC
# -------------------------
webrtc_streamer(
    key="ape_index_stream",
    video_processor_factory=PoseProcessor,
    media_stream_constraints={
        "video":{"frameRate":{"ideal":30}}, 
        "audio":False
    },
)
