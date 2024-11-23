import cv2
import mediapipe as mp

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

    results = pose.process(image_rgb)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks is not None:

        mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        hit_result = detect_hit(results.pose_landmarks.landmark)

        if hit_result is not None:
            cv2.putText(image_bgr, f"{hit_result}", (20, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)


    cv2.imshow('hit detection', image_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
