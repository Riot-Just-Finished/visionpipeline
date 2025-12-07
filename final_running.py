import cv2
import mediapipe as mp
import numpy as np
import time
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """Calculates the angle at point 'b'."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def smart_resize(frame, max_w=1200, max_h=900):
    """Resizes frame to fit on screen."""
    h, w = frame.shape[:2]
    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h)
    if scale >= 1.0:
        return frame
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_mode(landmarks):
    """
    Auto-detect Push-up or Pull-up. Ignores Squats.
    """
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    wrist_y = (landmarks[15].y + landmarks[16].y) / 2

    # Rough heuristic:
    if abs(shoulder_y - hip_y) < 0.25:
        return "PUSH-UP"
    else:
        if wrist_y < shoulder_y:
            return "PULL-UP"
        else:
            return "IDLE"


def main():
    print("---------------------------------------")
    print(" VIDEO SOURCE SELECTION")
    print("---------------------------------------")
    user_choice = input("Type 'w' for Webcam or 'v' for Video File: ").lower().strip()

    # ===== VIDEO SOURCE SELECTION =====
    source = 0  # default

    if user_choice == 'w':
        print("\nWebcam mode selected.")
        print("Tip: Usually 0 = internal cam, 1/2 = external cams.")
        cam_idx_str = input("Enter webcam index (press Enter for 0): ").strip()

        if cam_idx_str == "":
            source = 0
        else:
            if cam_idx_str.isdigit():
                source = int(cam_idx_str)
            else:
                print("Invalid input. Falling back to default webcam (0).")
                source = 0

        # Try opening selected camera; if it fails, fallback to 0
        test_cap = cv2.VideoCapture(source)
        if not test_cap.isOpened():
            print(f"Could not open camera index {source}. Falling back to 0.")
            test_cap.release()
            source = 0
        else:
            test_cap.release()

    elif user_choice == 'v':
        path = input("Enter video path: ").strip().replace('"', '').replace("'", "")
        if os.path.exists(path):
            source = path
        else:
            print("File not found. Using Webcam 0.")
            source = 0

    cap = cv2.VideoCapture(source)
    window_name = 'AI Trainer - Upper Body Only'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # ===== EXERCISE MODE SELECTION =====
    print("\n---------------------------------------")
    print(" EXERCISE MODE")
    print("---------------------------------------")
    print("1 - Push-ups only")
    print("2 - Pull-ups only")
    print("3 - Auto-detect (experimental)")
    mode_choice = input("Choose 1/2/3 (default = 1): ").strip()

    manual_mode = None
    if mode_choice == "2":
        manual_mode = "PULL-UP"
    elif mode_choice == "3":
        manual_mode = None      # use get_mode()
    else:
        manual_mode = "PUSH-UP" # default

    push_state = 0
    push_count = 0
    pull_state = 0
    pull_count = 0

    mode = "Detecting..."
    feedback_pct = 0.0

    # Pull-up thresholds for nose vs hands (bar)
    # offset = nose_y - hand_y
    # - bottom: nose clearly below hands (offset ~ positive)
    # - top:    nose slightly above hands (offset ~ negative / small)
    PULL_BOTTOM_MARGIN = 0.04   # hanging
    PULL_TOP_MARGIN = -0.02     # nose above hands

    with mp_pose.Pose(min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = smart_resize(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # 1. Mode selection
                if manual_mode is not None:
                    mode = manual_mode
                else:
                    mode = get_mode(lm)

                # 2. Elbow angles (for push-up depth)
                l_el = calculate_angle(
                    [lm[11].x, lm[11].y],
                    [lm[13].x, lm[13].y],
                    [lm[15].x, lm[15].y]
                )
                r_el = calculate_angle(
                    [lm[12].x, lm[12].y],
                    [lm[14].x, lm[14].y],
                    [lm[16].x, lm[16].y]
                )
                avg_elbow = (l_el + r_el) / 2.0

                feedback_pct = 0.0  # reset each frame

                # 3. Counting Logic
                # --- PUSH-UPS ---
                if mode == "PUSH-UP":
                    # Down < 90, Up > 160
                    # At bottom (90°) → depth 1, at top (160°) → depth 0
                    feedback_pct = float(np.interp(avg_elbow, (90, 160), (1, 0)))
                    feedback_pct = float(np.clip(feedback_pct, 0.0, 1.0))

                    if avg_elbow <= 90:
                        push_state = 1
                    if avg_elbow >= 160 and push_state == 1:
                        push_count += 1
                        push_state = 0
                        print(f"[PUSH-UP] Rep completed! Total: {push_count}")

                # --- PULL-UPS ---
                elif mode == "PULL-UP":
                    # Rep counted when NOSE goes above the HANDS (bar)
                    nose_y = lm[0].y
                    left_wrist_y = lm[15].y
                    right_wrist_y = lm[16].y

                    # Assume hands are holding the bar; take the higher hand (smaller y)
                    hand_y = min(left_wrist_y, right_wrist_y)

                    # Positive offset => nose below hands; negative => nose above
                    offset = nose_y - hand_y

                    # Rep depth bar:
                    #   offset >= PULL_BOTTOM_MARGIN: depth → 0 (hanging)
                    #   offset <= PULL_TOP_MARGIN:    depth → 1 (nose well above hands)
                    denom = PULL_BOTTOM_MARGIN - PULL_TOP_MARGIN
                    if denom > 1e-6:
                        depth = (PULL_BOTTOM_MARGIN - offset) / denom
                    else:
                        depth = 0.0
                    depth = float(np.clip(depth, 0.0, 1.0))
                    feedback_pct = depth

                    # State machine using same margins:
                    # Ready for new rep when fully hanging
                    if offset > PULL_BOTTOM_MARGIN:
                        pull_state = 0

                    # Count rep when nose clearly above hands
                    if (offset < PULL_TOP_MARGIN) and (pull_state == 0):
                        pull_count += 1
                        pull_state = 1
                        print(f"[PULL-UP] Rep completed! Total: {pull_count}")

                # --- IDLE ---
                elif mode == "IDLE":
                    feedback_pct = 0.0

                # 4. Drawing
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # UI Setup
                h, w, _ = image.shape
                ui_w = min(350, w)
                cv2.rectangle(image, (0, 0), (ui_w, 200), (30, 30, 30), -1)
                cv2.rectangle(image, (0, 0), (ui_w, 200), (0, 255, 0), 1)

                # Text Info
                cv2.putText(
                    image, f"MODE: {mode}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2
                )

                # Progress Bar
                cv2.putText(
                    image, "Rep Depth:",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1
                )
                cv2.rectangle(image, (10, 75), (ui_w - 20, 90), (100, 100, 100), -1)

                bar_color = (0, 0, 255)  # Red
                if feedback_pct > 0.95:
                    bar_color = (0, 255, 0)      # Green
                elif feedback_pct > 0.5:
                    bar_color = (0, 255, 255)    # Yellow

                bar_len = int((ui_w - 40) * feedback_pct)
                cv2.rectangle(image, (10, 75), (10 + bar_len, 90), bar_color, -1)

                # Counts (grey out inactive ones)
                c_p = (255, 255, 255) if mode == "PUSH-UP" else (80, 80, 80)
                c_u = (255, 255, 255) if mode == "PULL-UP" else (80, 80, 80)

                cv2.putText(
                    image, f"Push-ups: {int(push_count)}",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    c_p, 1
                )
                cv2.putText(
                    image, f"Pull-ups: {int(pull_count)}",
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    c_u, 1
                )

            cv2.imshow(window_name, image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
