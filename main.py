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
pose = mp_pose.Pose()

cap = cv2.VideoCapture('test/attack_test6.mov')  # замените на нужный файл

hit_display_time = 5
last_hit_time = 0
last_hit_type = None
hit_distance_threshold = 100

def detect_hit(landmarks):
    if landmarks:
        wrist_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        wrist_right = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        elbow_left = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        elbow_right = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

        if wrist_left.y > shoulder_left.y and wrist_right.y > shoulder_right.y and wrist_left.y > hip_left.y and wrist_right.y > hip_right.y:
            return "Serve"
        if abs(wrist_left.y - wrist_right.y) < 0.1 and abs(wrist_left.x - wrist_right.x) < 0.08 and abs(elbow_right.y - wrist_right.y) < 0.05:
            return "Lower hit"
        elif (wrist_left.y < ear.y and wrist_right.y > shoulder_right.y) or (wrist_left.y > ear.y and wrist_right.y < shoulder_right.y):
            return 'Attack'
        elif elbow_left.y < shoulder_left.y and elbow_right.y < shoulder_right.y:
            return 'Upper hit'

    return None

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([20, 155, 100])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_center = None
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        if radius > 10:
            ball_center = (int(x), int(y))
            cv2.circle(frame, ball_center, int(radius), (0, 255, 0), 2)
            cv2.circle(frame, ball_center, 5, (255, 0, 0), -1)

    h, w, _ = frame.shape
    person_1 = frame[:, :w // 2]
    person_2 = frame[:, w // 2:]

    image_1_rgb = cv2.cvtColor(person_1, cv2.COLOR_BGR2RGB)
    results_1 = pose.process(image_1_rgb)
    landmarks_1 = results_1.pose_landmarks.landmark if results_1.pose_landmarks else None

    image_2_rgb = cv2.cvtColor(person_2, cv2.COLOR_BGR2RGB)
    results_2 = pose.process(image_2_rgb)
    landmarks_2 = results_2.pose_landmarks.landmark if results_2.pose_landmarks else None

    closest_player = None
    closest_distance = float('inf')

    if ball_center:
        if landmarks_1:
            wrist_left_1 = (int(landmarks_1[mp_pose.PoseLandmark.LEFT_WRIST].x * w // 2), int(landmarks_1[mp_pose.PoseLandmark.LEFT_WRIST].y * h))
            wrist_right_1 = (int(landmarks_1[mp_pose.PoseLandmark.RIGHT_WRIST].x * w // 2), int(landmarks_1[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))
            distance_1 = min(distance(ball_center, wrist_left_1), distance(ball_center, wrist_right_1))
        else:
            distance_1 = float('inf')

        if landmarks_2:
            wrist_left_2 = (int(landmarks_2[mp_pose.PoseLandmark.LEFT_WRIST].x * w // 2 + w // 2), int(landmarks_2[mp_pose.PoseLandmark.LEFT_WRIST].y * h))
            wrist_right_2 = (int(landmarks_2[mp_pose.PoseLandmark.RIGHT_WRIST].x * w // 2 + w // 2), int(landmarks_2[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))
            distance_2 = min(distance(ball_center, wrist_left_2), distance(ball_center, wrist_right_2))
        else:
            distance_2 = float('inf')

        if landmarks_1 and landmarks_2:
            if distance_1 < distance_2:
                closest_distance = distance_1
                closest_player = 1
            else:
                closest_distance = distance_2
                closest_player = 2
        elif landmarks_1:
            closest_player = 1
            closest_distance = distance_1
        elif landmarks_2:
            closest_player = 2
            closest_distance = distance_2

    if closest_player is None:
        closest_player = 1 if landmarks_1 else 2

    hit_result = None
    if closest_distance < hit_distance_threshold:
        if closest_player == 1 and landmarks_1:
            hit_result = detect_hit(landmarks_1)
        elif closest_player == 2 and landmarks_2:
            hit_result = detect_hit(landmarks_2)

    if hit_result:
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
        cv2.putText(frame, last_hit_type, (20, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    cv2.imshow('Hit Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



