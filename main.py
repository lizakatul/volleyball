import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pose = mp_pose.Pose()

def detect_hit(landmarks):
    if landmarks:
        wrist_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        wrist_right = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow_left = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        elbow_right = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        if abs(wrist_left.y - wrist_right.y) < 0.05 and abs(wrist_left.x - wrist_right.x) < 0.1 and abs(elbow_left.y - elbow_right.y) < 0.05 and abs(elbow_left.x - elbow_right.x) < 0.1 and wrist_left.y > shoulder_left.y and wrist_right.y > shoulder_right.y:
            return "Lower hit"

        if wrist_left.y < shoulder_left.y and wrist_right.y < shoulder_right.y:
            return 'Upper hit'

        elif wrist_left.y < ear.y and wrist_right.y > shoulder_right.y:
            return 'Attack'

        return None

    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    blue_lower = np.array([100, 150, 50])
    blue_upper = np.array([140, 255, 255])

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    combined_mask = cv2.bitwise_or(yellow_mask, blue_mask)

    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, structure)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, structure)§

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Center: ({center[0]}, {center[1]})", (center[0]+10, center[1]), cv2.FONT_ITALIC,0.5, (255, 255, 255), 2)

    results = pose.process(image_rgb)

    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        hit_result = detect_hit(results.pose_landmarks.landmark)

        if hit_result is not None:
            cv2.putText(frame, f"{hit_result}", (20, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    cv2.imshow('hit detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
