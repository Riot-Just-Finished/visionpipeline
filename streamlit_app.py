import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
# Removing GRUModel from import as it was deleted in final_running.py
from final_running import get_mode, calculate_angle, smart_resize, mp_pose, mp_drawing

st.set_page_config(page_title="AI Trainer", layout="wide")

st.sidebar.title("Configuration")

# Video Source Selection
source_type = st.sidebar.radio("Select Video Source", ["Webcam", "Video File"])

source = 0
if source_type == "Webcam":
    cam_index = st.sidebar.number_input("Webcam Index", min_value=0, value=0, step=1)
    source = int(cam_index)
else:
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        source = "temp_video.mp4"

# Exercise Mode Selection
st.sidebar.subheader("Exercise Mode")
mode_choice = st.sidebar.selectbox(
    "Choose Mode",
    ["Push-ups only", "Pull-ups only", "Auto-detect (experimental)"],
    index=0
)

manual_mode = None
if mode_choice == "Pull-ups only":
    manual_mode = "PULL-UP"
elif mode_choice == "Auto-detect (experimental)":
    manual_mode = None  # Use get_mode()
else:
    manual_mode = "PUSH-UP"

# Start/Stop Buttons
st.sidebar.subheader("Controls")
if 'training_active' not in st.session_state:
    st.session_state['training_active'] = False

col_start, col_stop = st.sidebar.columns(2)
with col_start:
    if st.button("Start Training"):
        st.session_state['training_active'] = True
with col_stop:
    if st.button("Stop Training"):
        st.session_state['training_active'] = False

st.title("AI Trainer - Upper Body Specialist")

# Layout: Video on Left, Metrics on Right
col1, col2 = st.columns([2, 1])

with col1:
    st_frame = st.empty()

with col2:
    st.subheader("Metrics")
    
    # Metrics placeholders
    mode_text = st.empty()
    push_metric = st.empty()
    pull_metric = st.empty()
    feedback_bar = st.empty()
    feedback_text = st.empty()

    # Initial values
    mode_text.markdown(f"**Mode:** Waiting...")
    push_metric.metric("Push-ups", 0)
    pull_metric.metric("Pull-ups", 0)
    feedback_text.text("Rep Depth: 0%")
    feedback_bar.progress(0)

if st.session_state['training_active']:
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        st.error(f"Error opening video source: {source}")
        st.session_state['training_active'] = False
    else:
        # State Variables
        push_state = 0; push_count = 0
        pull_state = 0; pull_count = 0
        mode = "Detecting..."
        feedback_pct = 0.0
        
        # Pull-up thresholds from final_running.py
        PULL_BOTTOM_MARGIN = 0.04   # hanging
        PULL_TOP_MARGIN = -0.02     # nose above hands
        
        with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
            while cap.isOpened() and st.session_state['training_active']:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Video finished or stream ended.")
                    st.session_state['training_active'] = False
                    break
                
                # Resize and Process
                frame = sr_frame = smart_resize(frame)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # 1. Mode Selection
                    if manual_mode is not None:
                        mode = manual_mode
                    else:
                        mode = get_mode(lm)
    
                    # 2. Elbow Angles
                    l_el = calculate_angle([lm[11].x,lm[11].y], [lm[13].x,lm[13].y], [lm[15].x,lm[15].y])
                    r_el = calculate_angle([lm[12].x,lm[12].y], [lm[14].x,lm[14].y], [lm[16].x,lm[16].y])
                    avg_elbow = (l_el + r_el) / 2.0
                    
                    # Reset feedback
                    feedback_pct = 0.0

                    # 3. Counting Logic
                    # --- PUSH-UPS ---
                    if mode == "PUSH-UP":
                        # Down < 90, Up > 160
                        feedback_pct = float(np.interp(avg_elbow, (90, 160), (1, 0)))
                        feedback_pct = float(np.clip(feedback_pct, 0.0, 1.0))
                        
                        if avg_elbow <= 90: 
                            push_state = 1
                        if avg_elbow >= 160 and push_state == 1: 
                            push_count += 1
                            push_state = 0
    
                    # --- PULL-UPS ---
                    elif mode == "PULL-UP":
                        nose_y = lm[0].y
                        left_wrist_y = lm[15].y
                        right_wrist_y = lm[16].y
                        
                        # Assume hands are holding the bar; take the higher hand (smaller y)
                        hand_y = min(left_wrist_y, right_wrist_y)
                        
                        # Positive offset => nose below hands; negative => nose above
                        offset = nose_y - hand_y
                        
                        # Rep depth bar
                        denom = PULL_BOTTOM_MARGIN - PULL_TOP_MARGIN
                        if denom > 1e-6:
                            depth = (PULL_BOTTOM_MARGIN - offset) / denom
                        else:
                            depth = 0.0
                        
                        depth = float(np.clip(depth, 0.0, 1.0))
                        feedback_pct = depth
                        
                        # State machine
                        if offset > PULL_BOTTOM_MARGIN:
                            pull_state = 0
                        
                        if (offset < PULL_TOP_MARGIN) and (pull_state == 0):
                            pull_count += 1
                            pull_state = 1
                    
                    # --- IDLE ---
                    elif mode == "IDLE":
                        feedback_pct = 0.0
    
                    # 4. Drawing
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Update UI
                st_frame.image(image, channels="RGB", use_container_width=True)
                
                # Update Metrics
                mode_color = "green" if mode != "IDLE" else "gray"
                mode_text.markdown(f"**Mode:** <span style='color:{mode_color}'>{mode}</span>", unsafe_allow_html=True)
                
                push_metric.metric("Push-ups", int(push_count))
                pull_metric.metric("Pull-ups", int(pull_count))
                
                feedback_bar.progress(min(max(float(feedback_pct), 0.0), 1.0))
                feedback_text.text(f"Rep Depth: {int(feedback_pct * 100)}%")

        cap.release()
else:
    st.info("Click 'Start Training' to begin.")
