import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

gaze_start_time = None
face_start_time = None
face_not_detected_start_time = None
face_down_count = 0
gaze_left_count = 0
gaze_right_count = 0

gaze_warning_shown = False
face_warning_shown = False
face_not_detected_warning_shown = False
face_down_warning_shown = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)

    face_detected = False
    face_direction = "Center"
    gaze_direction = "Center"

    if face_results.detections:
        face_detected = True
        face_not_detected_start_time = None
        face_not_detected_warning_shown = False
    else:
        if face_not_detected_start_time is None:
            face_not_detected_start_time = time.time()
        elif time.time() - face_not_detected_start_time > 15 and not face_not_detected_warning_shown:
            print("⚠️ WARNING: Face not detected for 15 seconds!")
            face_not_detected_warning_shown = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            chin = face_landmarks.landmark[152]

            nose_x = int(nose.x * w)
            left_eye_x = int(left_eye.x * w)
            right_eye_x = int(right_eye.x * w)
            chin_y = int(chin.y * h)

            if nose_x < left_eye_x:
                face_direction = "Looking Right"
            elif nose_x > right_eye_x:
                face_direction = "Looking Left"
            elif chin_y > h * 0.75:
                face_direction = "Looking Down"
                face_down_count += 1
            else:
                face_direction = "Center"
                face_warning_shown = False


            if (face_direction == "Looking Right" or face_direction == "Looking Left") and not face_warning_shown:
                if face_start_time is None:
                    face_start_time = time.time()
                elif time.time() - face_start_time > 5:
                    print(f" WARNING: Face turned {face_direction} for too long!")
                    face_warning_shown = True
            else:
                face_start_time = None

            if face_down_count >= 4 and not face_down_warning_shown:
                print(" WARNING: Looking down multiple times!")
                face_down_warning_shown = True


            eye_x_avg = (left_eye_x + right_eye_x) // 2
            if eye_x_avg < w * 0.4:
                gaze_direction = "Looking Right"
                gaze_right_count += 1
            elif eye_x_avg > w * 0.6:
                gaze_direction = "Looking Left"
                gaze_left_count += 1
            else:
                gaze_direction = "Center"
                gaze_warning_shown = False


            if gaze_direction == "Looking Left" and gaze_left_count >= 10 and not gaze_warning_shown:
                print(" WARNING: Looking left for too long!")
                gaze_warning_shown = True
            elif gaze_direction == "Looking Right" and gaze_right_count >= 10 and not gaze_warning_shown:
                print(" WARNING: Looking right for too long!")
                gaze_warning_shown = True


            if gaze_direction == "Center":
                gaze_left_count = 0
                gaze_right_count = 0


            cv2.circle(frame, (nose_x, int(nose.y * h)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (left_eye_x, int(left_eye.y * h)), 5, (255, 0, 0), -1)
            cv2.circle(frame, (right_eye_x, int(right_eye.y * h)), 5, (255, 0, 0), -1)


    cv2.putText(frame, f"Face: {face_direction}", (30, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Gaze: {gaze_direction}", (30, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)


    cv2.imshow("CAUGHT", frame)


    if cv2.waitKey(1)&0xFF ==27:
        break

cap.release()
cv2.destroyAllWindows()
