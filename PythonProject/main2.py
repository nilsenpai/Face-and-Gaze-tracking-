import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark indices
LEFT_PUPIL = 468
RIGHT_PUPIL = 473
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Accuracy Tracking
total_detections = 0
correct_detections = 0

# Define ground truth manually (Modify as per test cases)
# Example: {"Looking Left": ["LEFT"], "Looking Right": ["RIGHT"], "Looking Center": ["CENTER"]}
GROUND_TRUTH = {
    "LEFT": "Looking Left",
    "RIGHT": "Looking Right",
    "CENTER": "Looking Center"
}

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with face mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get key points
            left_pupil = face_landmarks.landmark[LEFT_PUPIL]
            right_pupil = face_landmarks.landmark[RIGHT_PUPIL]
            left_eye_inner = face_landmarks.landmark[LEFT_EYE_INNER]
            left_eye_outer = face_landmarks.landmark[LEFT_EYE_OUTER]
            right_eye_inner = face_landmarks.landmark[RIGHT_EYE_INNER]
            right_eye_outer = face_landmarks.landmark[RIGHT_EYE_OUTER]

            # Convert to pixel coordinates
            left_pupil_x = int(left_pupil.x * w)
            right_pupil_x = int(right_pupil.x * w)
            left_eye_inner_x = int(left_eye_inner.x * w)
            left_eye_outer_x = int(left_eye_outer.x * w)
            right_eye_inner_x = int(right_eye_inner.x * w)
            right_eye_outer_x = int(right_eye_outer.x * w)

            # Compute pupil displacement ratio
            left_eye_width = left_eye_inner_x - left_eye_outer_x
            right_eye_width = right_eye_outer_x - right_eye_inner_x

            left_pupil_ratio = (left_pupil_x - left_eye_outer_x) / left_eye_width
            right_pupil_ratio = (right_pupil_x - right_eye_inner_x) / right_eye_width

            # Determine eye movement
            if left_pupil_ratio < 0.38 and right_pupil_ratio < 0.38:
                eye_direction = "Looking Left"
            elif left_pupil_ratio > 0.62 and right_pupil_ratio > 0.62:
                eye_direction = "Looking Right"
            elif 0.40 <= left_pupil_ratio <= 0.60 and 0.40 <= right_pupil_ratio <= 0.60:
                eye_direction = "Looking Center"
            else:
                eye_direction = "Unknown"

            # Accuracy Calculation
            total_detections += 1
            user_input = input(f"Is this correct? {eye_direction} (yes/no): ").strip().lower()

            if user_input == "yes":
                correct_detections += 1

            # Draw Debugging Circles
            cv2.circle(frame, (left_pupil_x, int(left_pupil.y * h)), 3, (0, 255, 0), -1)
            cv2.circle(frame, (right_pupil_x, int(right_pupil.y * h)), 3, (0, 255, 0), -1)

            # Display direction
            cv2.putText(frame, eye_direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow('AI Cheating Detector', frame)

    # Exit on 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Compute Accuracy
accuracy = (correct_detections / total_detections) * 100 if total_detections > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")

cap.release()
cv2.destroyAllWindows()
