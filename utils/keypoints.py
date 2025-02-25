import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def classify_gait(left_knee_angle):
    """Classify gait based on knee angle."""
    if left_knee_angle < 140:
        return "Limping"
    elif left_knee_angle > 170:
        return "No Arm Swing"
    else:
        return "Normal"

def extract_keypoints(video_path):
    """Extracts pose keypoints from an MP4 video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    keypoints_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
            left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            gait_class = classify_gait(left_knee_angle)

            keypoints_data.append({
                "frame": len(keypoints_data) + 1,
                "left_knee_angle": left_knee_angle,
                "gait_class": gait_class
            })

    cap.release()
    return keypoints_data
