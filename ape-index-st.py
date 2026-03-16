import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import time
import math
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
MARKER_SIZE_M = 0.10
MARKER_TIMEOUT = 0.3
APEX_MEAN = 1.0
APEX_SD = 0.05

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT)

# -------------------------
# Thread-safe flags
# -------------------------
class SharedFlags:
    def __init__(self):
        self.mirror = False
        self.unfreeze = False
        self.is_frozen = False
        self.frozen_frame = None
        self.pose_start_time = None
        self.pixels_per_meter = None
        self.last_marker_time = 0
        self.show_frozen_text = False
        self.lock = threading.Lock()

if 'shared_flags' not in st.session_state:
    st.session_state['shared_flags'] = SharedFlags()

shared_flags = st.session_state['shared_flags']

# -------------------------
st.title("Ape Index Measurement")
st.write("Hold a T-pose to measure your ape index. Place an ArUco marker (4x4_50, 100mm) in view for accurate scaling.")

# Buttons
# -------------------------
with st.sidebar:
    st.title("Controls")
    if st.button("Mirror Webcam", help="Flip the webcam image horizontally"):
        with shared_flags.lock:
            shared_flags.mirror = not shared_flags.mirror
    if st.button("Unfreeze", help="Resume live video feed"):
        with shared_flags.lock:
            shared_flags.unfreeze = True

# -------------------------
# Helpers
# -------------------------
def to_pixel(landmark, width, height):
    return np.array([landmark.x*width, landmark.y*height])

def distance(a,b):
    return np.linalg.norm(a-b)

def is_t_pose(lm_px, img_height, mirror=False, tol_ratio=0.1):
    tol_px = img_height * tol_ratio
    l_sh = lm_px[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm_px[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_wr = lm_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wr = lm_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    if mirror:
        l_sh, r_sh = r_sh, l_sh
        l_wr, r_wr = r_wr, l_wr
    return abs(l_sh[1]-l_wr[1])<tol_px and abs(r_sh[1]-r_wr[1])<tol_px

def detect_aruco(frame_bgr, display_frame, scale_x, scale_y):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        c = corners[0][0]
        width_px = np.linalg.norm(c[0]-c[1])
        c_scaled = c * [scale_x, scale_y]
        cv2.polylines(display_frame, [c_scaled.astype(int)], True, (255,255,0), 2)
        with shared_flags.lock:
            shared_flags.last_marker_time = time.time()
            shared_flags.pixels_per_meter = width_px / MARKER_SIZE_M
    else:
        with shared_flags.lock:
            if time.time() - shared_flags.last_marker_time > MARKER_TIMEOUT:
                shared_flags.pixels_per_meter = None

def draw_bell_curve(frame, ape_index):
    h, w, _ = frame.shape
    curve_w, curve_h = 200, 100
    start_x, start_y = w - curve_w - 20, h - curve_h - 20
    x_vals = np.linspace(APEX_MEAN - 3*APEX_SD, APEX_MEAN + 3*APEX_SD, curve_w)
    pdf = 1/(APEX_SD*np.sqrt(2*np.pi))*np.exp(-0.5*((x_vals-APEX_MEAN)/APEX_SD)**2)
    pdf_scaled = (pdf/pdf.max()*curve_h).astype(int)
    pts = [(start_x+i, start_y+curve_h-pdf_scaled[i]) for i in range(curve_w)]
    for i in range(len(pts)-1):
        cv2.line(frame, pts[i], pts[i+1], (180,180,255), 2)
    rel_pos = max(0.0, min(1.0, (ape_index-(APEX_MEAN-3*APEX_SD))/(6*APEX_SD)))
    ape_x = int(rel_pos*curve_w)+start_x
    cv2.line(frame, (ape_x, start_y), (ape_x, start_y+curve_h), (0,0,255), 2)
    percentile = int((0.5*(1+math.erf((ape_index-APEX_MEAN)/(APEX_SD*math.sqrt(2)))))*100)
    cv2.putText(frame, f"{percentile}th percentile", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)

# -------------------------
# Video Processor
# -------------------------
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)
        self.ape_buffer = deque(maxlen=BUFFER_SIZE)
        self.arm_buffer = deque(maxlen=BUFFER_SIZE)
        self.height_buffer = deque(maxlen=BUFFER_SIZE)

    def recv(self, frame):
        # Thread-safe flags
        with shared_flags.lock:
            mirror_flag = shared_flags.mirror
            unfreeze_flag = shared_flags.unfreeze
            is_frozen = shared_flags.is_frozen
            frozen_frame = shared_flags.frozen_frame
            pose_start_time = shared_flags.pose_start_time
            if unfreeze_flag:
                shared_flags.unfreeze = False

        # Unfreeze
        if unfreeze_flag:
            with shared_flags.lock:
                shared_flags.is_frozen = False
                shared_flags.frozen_frame = None
                shared_flags.pose_start_time = None

        # Convert frame and resize for display
        img = frame.to_ndarray(format="bgr24")
        if mirror_flag:
            img = cv2.flip(img, 1)

        display_frame = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        scale_x = DISPLAY_WIDTH / img.shape[1]
        scale_y = DISPLAY_HEIGHT / img.shape[0]

        # Detect ArUco
        detect_aruco(img, display_frame, scale_x, scale_y)

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
                cv2.line(display_frame, (x1, y1), (x2, y2), (255,255,255), 2)

            # T-pose detection
            lm_px = {i: to_pixel(lm, DISPLAY_WIDTH, DISPLAY_HEIGHT) for i, lm in enumerate(result.pose_landmarks.landmark)}
            if is_t_pose(lm_px, DISPLAY_HEIGHT, mirror_flag):
                t_pose_detected = True
                l_wrist = lm_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
                r_wrist = lm_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                arm_span_px = distance(l_wrist,r_wrist)
                head_y = lm_px[mp_pose.PoseLandmark.NOSE.value][1]
                l_ankle = lm_px[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                r_ankle = lm_px[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                ankle_mid = (l_ankle + r_ankle)/2
                height_px = ankle_mid[1] - head_y

                if height_px>0:
                    ape_index = arm_span_px/height_px
                    self.ape_buffer.append(ape_index)
                    self.arm_buffer.append(arm_span_px)
                    self.height_buffer.append(height_px)
                    ape_index_avg = np.mean(self.ape_buffer)

                    # Draw lines
                    cv2.line(display_frame, tuple(l_wrist.astype(int)), tuple(r_wrist.astype(int)), (255,0,0), 3)
                    cv2.line(display_frame, (int((l_wrist[0]+r_wrist[0])/2), int(head_y)), tuple(ankle_mid.astype(int)), (0,0,255), 3)

                    # Overlay text
                    cv2.putText(display_frame, f"Ape Index: {ape_index_avg:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    with shared_flags.lock:
                        ppm = shared_flags.pixels_per_meter
                    if ppm:
                        h_m = height_px / ppm
                        a_m = arm_span_px / ppm
                        cv2.putText(display_frame, f"Height: {h_m:.2f} m", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        cv2.putText(display_frame, f"Arm Span: {a_m:.2f} m", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    else:
                        cv2.putText(display_frame, f"Height: {height_px:.1f} px", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        cv2.putText(display_frame, f"Arm Span: {arm_span_px:.1f} px", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                    # Draw bell curve
                    draw_bell_curve(display_frame, ape_index_avg)

        # Freeze logic
        if t_pose_detected:
            with shared_flags.lock:
                if shared_flags.pose_start_time is None:
                    shared_flags.pose_start_time = time.time()
                elif (time.time()-shared_flags.pose_start_time) >= 1.0 and not shared_flags.show_frozen_text:
                    shared_flags.show_frozen_text = True
                    shared_flags.frozen_frame = display_frame.copy()
                elif (time.time()-shared_flags.pose_start_time) >= FREEZE_DURATION:
                    shared_flags.is_frozen = True
        else:
            with shared_flags.lock:
                shared_flags.pose_start_time = None
                shared_flags.show_frozen_text = False

        with shared_flags.lock:
            if shared_flags.is_frozen and shared_flags.frozen_frame is not None:
                display_frame = shared_flags.frozen_frame.copy()

        display_frame = np.ascontiguousarray(display_frame, dtype=np.uint8)

        # Overlay text
        with shared_flags.lock:
            if shared_flags.show_frozen_text or shared_flags.is_frozen:
                cv2.putText(display_frame, "*Frozen*", (DISPLAY_WIDTH-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if mirror_flag:
            cv2.putText(display_frame, "Mirror ON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")

# -------------------------
# Start WebRTC
# -------------------------
webrtc_streamer(
    key="ape_index_stream",
    video_processor_factory=PoseProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1920},
            "height": {"ideal": 1080},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
)
