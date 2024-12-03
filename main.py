import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

pygame.mixer.init()
upper_sound = pygame.mixer.Sound('sounds/Upper hit.mp3')
lower_sound = pygame.mixer.Sound('sounds/Lower hit.mp3')
serve_sound = pygame.mixer.Sound('sounds/Serve.mp3')
attack_sound = pygame.mixer.Sound('sounds/Attack.mp3')

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('test/attack_test4.mov')  #заменить на нужный файл

pose = mp_pose.Pose()
hit_display_time = 2
last_hit_time = 0
last_hit_type = None

def detect_hit(landmarks):
    if landmarks:
        wrist_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        wrist_right = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow_left = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        elbow_right = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        if wrist_left.y > shoulder_left.y and wrist_right.y > shoulder_right.y and wrist_left.y > hip_left.y and wrist_right.y > hip_right.y:
            return "Serve"

        if abs(wrist_left.y - wrist_right.y) < 0.02 and abs(wrist_left.x - wrist_right.x) < 0.02 and abs(elbow_left.y - elbow_right.y) < 0.03 and abs(elbow_left.x - elbow_right.x) < 0.1:
            return "Lower hit"

        elif (wrist_left.y < ear.y and wrist_right.y > shoulder_right.y) or (wrist_left.y > ear.y  and wrist_right.y < shoulder_right.y):
            return 'Attack'

        elif wrist_left.y < shoulder_left.y and wrist_right.y < shoulder_right.y:
            return 'Upper hit'

        return None
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([20, 155, 100])
    yellow_upper = np.array([30, 255, 255])
    blue_lower = np.array([100, 150, 50])
    blue_upper = np.array([140, 255, 255])

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    combined_mask = cv2.bitwise_or(yellow_mask, blue_mask)

    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, structure)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, structure)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_center = None
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        if radius > 10:
            ball_center = (int(x), int(y))
            cv2.circle(frame, ball_center, int(radius), (0, 255, 0), 2)
            cv2.circle(frame, ball_center, 5, (255, 0, 0), -1)
            cv2.putText(frame, f"Ball: ({ball_center[0]}, {ball_center[1]})", (ball_center[0] + 10, ball_center[1]), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2)

    results = pose.process(image_rgb)

    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        hit_result = detect_hit(landmarks)
        if hit_result and ball_center:
            wrist_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            wrist_right = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            wrist_left_pos = (int(wrist_left.x * frame.shape[1]), int(wrist_left.y * frame.shape[0]))
            wrist_right_pos = (int(wrist_right.x * frame.shape[1]), int(wrist_right.y * frame.shape[0]))

            if (np.linalg.norm(np.array(wrist_left_pos) - np.array(ball_center)) < 50 or
                    np.linalg.norm(np.array(wrist_right_pos) - np.array(ball_center)) < 50):
                last_hit_time = time.time()
                last_hit_type = hit_result
                if last_hit_type == 'Upper hit':
                    upper_sound.play()
                elif last_hit_type == 'Lower hit':
                    lower_sound.play()
                elif last_hit_type == 'Attack':
                    attack_sound.play()
                elif last_hit_type == 'Serve':
                    serve_sound.play()

    if last_hit_type and (time.time() - last_hit_time < hit_display_time):
        cv2.putText(frame, f"{last_hit_type}", (20, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    cv2.imshow('hit detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


