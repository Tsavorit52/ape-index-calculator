import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from PIL import Image

# -------------------------
# Session state
# -------------------------
if "is_frozen" not in st.session_state:
    st.session_state.is_frozen = False
if "pose_start_time" not in st.session_state:
    st.session_state.pose_start_time = None
if "mirror" not in st.session_state:
    st.session_state.mirror = False
if "pixels_per_meter" not in st.session_state:
    st.session_state.pixels_per_meter = None
if "last_marker_time" not in st.session_state:
    st.session_state.last_marker_time = 0

# -------------------------
# Constants
# -------------------------
APEX_MEAN = 1.0
APEX_SD = 0.05
BUFFER_SIZE = 20
FREEZE_DURATION = 1.5  # seconds to freeze
MARKER_SIZE_M = 0.10
MARKER_TIMEOUT = 0.3

ape_buffer = deque(maxlen=BUFFER_SIZE)
arm_buffer = deque(maxlen=BUFFER_SIZE)
height_buffer = deque(maxlen=BUFFER_SIZE)

last_ape_index = None
last_arm_span = None
last_height = None
frozen_frame = None

# -------------------------
# MediaPipe
# -------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# -------------------------
# ArUco
# -------------------------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT)

def detect_aruco(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        c = corners[0][0]
        width_px = np.linalg.norm(c[0]-c[1])
        cv2.polylines(frame_bgr, [c.astype(int)], True, (255,255,0), 2)
        st.session_state.last_marker_time = time.time()
        st.session_state.pixels_per_meter = width_px / MARKER_SIZE_M
        return True
    else:
        if time.time() - st.session_state.last_marker_time > MARKER_TIMEOUT:
            st.session_state.pixels_per_meter = None
        return False

# -------------------------
# Helper functions
# -------------------------
def to_pixel(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height])

def distance(a,b):
    return np.linalg.norm(a-b)

def is_t_pose_pixels(lm_px, tol_px=100):
    l_sh = lm_px[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm_px[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_wr = lm_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wr = lm_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    return abs(l_sh[1]-l_wr[1])<tol_px and abs(r_sh[1]-r_wr[1])<tol_px

def draw_bell_curve(frame, ape_index):
    h, w, _ = frame.shape
    curve_w, curve_h = 200, 100
    start_x, start_y = w - curve_w - 20, h - curve_h - 20

    x_vals = np.linspace(APEX_MEAN - 3*APEX_SD, APEX_MEAN + 3*APEX_SD, curve_w)
    pdf = 1/(APEX_SD * np.sqrt(2*np.pi)) * np.exp(-0.5*((x_vals - APEX_MEAN)/APEX_SD)**2)
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

    percentile = int((0.5*(1 + math.erf((ape_index - APEX_MEAN)/(APEX_SD*math.sqrt(2))))) * 100)
    cv2.putText(frame, f"{percentile}th percentile", (start_x, start_y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# -------------------------
# Detect cameras
# -------------------------
def list_cameras(max_tested=10):
    available = []
    names = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            name = f"Camera {i}"
            if cv2.__version__ >= "4.5":
                try:
                    desc = cap.get(cv2.CAP_PROP_DEVICE_DESCRIPTION)
                    if desc:
                        name = desc
                except:
                    pass
            names.append(name)
            cap.release()
    return available, names

cam_indices, cam_names = list_cameras()
if not cam_indices:
    st.error("No cameras detected!")
else:
    cam_choice = st.selectbox(
        "Select camera",
        options=cam_indices,
        format_func=lambda i: cam_names[cam_indices.index(i)]
    )
    st.write(f"Using: {cam_names[cam_indices.index(cam_choice)]}")

# -------------------------
# Buttons
# -------------------------
if st.button("Flip Image"):
    st.session_state.mirror = not st.session_state.mirror

if st.button("Unfreeze"):
    st.session_state.is_frozen = False
    st.session_state.pose_start_time = None

stframe = st.empty()
cap = cv2.VideoCapture(cam_choice)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# -------------------------
# Main loop
# -------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to read from camera")
        break

    # Marker detection
    detect_aruco(frame)

    # Convert BGR -> RGB & mirror
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if st.session_state.mirror:
        frame_rgb = cv2.flip(frame_rgb, 1)

    # MediaPipe pose
    result = pose.process(frame_rgb)
    t_pose_detected = False

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        lm_px = {idx: to_pixel(lm[idx], frame.shape[1], frame.shape[0]) for idx in range(len(lm))}
        mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if is_t_pose_pixels(lm_px):
            t_pose_detected = True

            # Arm span
            l_wrist = lm_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
            r_wrist = lm_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            l_index = lm_px[mp_pose.PoseLandmark.LEFT_INDEX.value]
            r_index = lm_px[mp_pose.PoseLandmark.RIGHT_INDEX.value]

            left_x = min(l_wrist[0], l_index[0])
            right_x = max(r_wrist[0], r_index[0])
            l_y = l_wrist[1]
            r_y = r_wrist[1]
            arm_span_px = distance(np.array([left_x,l_y]), np.array([right_x,r_y]))

            # Height from head to ankles
            head_y = min(lm_px[mp_pose.PoseLandmark.NOSE.value][1],
                         lm_px[mp_pose.PoseLandmark.LEFT_EAR.value][1],
                         lm_px[mp_pose.PoseLandmark.RIGHT_EAR.value][1])
            l_ankle = lm_px[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            r_ankle = lm_px[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            ankle_mid = (l_ankle + r_ankle)/2
            height_px = ankle_mid[1]-head_y

            if height_px>0:
                ape_index = arm_span_px / height_px
                ape_buffer.append(ape_index)
                arm_buffer.append(arm_span_px)
                height_buffer.append(height_px)

                last_ape_index = np.mean(ape_buffer)
                last_arm_span = np.mean(arm_buffer)
                last_height = np.mean(height_buffer)

                # Draw blue line full height
                cv2.line(frame_rgb, (int((left_x+right_x)/2), int(head_y)),
                         tuple(ankle_mid.astype(int)), (0,0,255), 3)
                # Draw arm span line
                cv2.line(frame_rgb, (int(left_x),int(l_y)), (int(right_x),int(r_y)), (255,0,0), 3)

    # Overlay text & bell curve BEFORE freeze
    display_frame = frame_rgb.copy()
    if last_ape_index is not None:
        if st.session_state.pixels_per_meter:
            height_m = last_height / st.session_state.pixels_per_meter
            arm_span_m = last_arm_span / st.session_state.pixels_per_meter
            cv2.putText(display_frame, f"Height: {height_m:.2f} m", (30,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(display_frame, f"Arm span: {arm_span_m:.2f} m", (30,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        else:
            cv2.putText(display_frame, f"Height: {int(last_height)} px", (30,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(display_frame, f"Arm span: {int(last_arm_span)} px", (30,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.putText(display_frame, f"Ape Index: {last_ape_index:.2f}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        draw_bell_curve(display_frame, last_ape_index)

    # -------------------------
    # Freeze logic (T-pose only)
    # -------------------------
    if t_pose_detected:
        if st.session_state.pose_start_time is None:
            st.session_state.pose_start_time = time.time()
        elif (time.time()-st.session_state.pose_start_time) >= FREEZE_DURATION:
            frozen_frame = display_frame.copy()
            st.session_state.is_frozen = True
    else:
        st.session_state.pose_start_time = None

    final_frame = frozen_frame if st.session_state.is_frozen else display_frame

    # Draw asterisk when frozen
    if st.session_state.is_frozen:
        h, w, _ = final_frame.shape
        cv2.putText(final_frame, "*", (w-40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Display video
    stframe.image(final_frame, channels="RGB", width='stretch')
