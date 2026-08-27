import cv2
import mediapipe as mp
import math
import time
import winsound
def calculate_ear(landmarks, eye_indices):

    # Get the 6 eye points
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    # Vertical distances
    vertical_1 = math.dist(
        (p2.x, p2.y),
        (p6.x, p6.y)
    )

    vertical_2 = math.dist(
        (p3.x, p3.y),
        (p5.x, p5.y)
    )

    # Horizontal distance
    horizontal = math.dist(
        (p1.x, p1.y),
        (p4.x, p4.y)
    )

    # Eye Aspect Ratio
    ear = (vertical_1 + vertical_2) / (2 * horizontal)

    return ear
def calculate_mar(landmarks):

    left_corner = landmarks[61]
    right_corner = landmarks[291]

    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    vertical = math.dist(
        (upper_lip.x, upper_lip.y),
        (lower_lip.x, lower_lip.y)
    )

    horizontal = math.dist(
        (left_corner.x, left_corner.y),
        (right_corner.x, right_corner.y)
    )

    mar = vertical / horizontal

    return mar
FACE_DETECTOR_MODEL = "models/blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = "models/face_landmarker.task"


# ==============================
# FACE DETECTOR
# ==============================

face_detector = mp.tasks.vision.FaceDetector.create_from_model_path(
    FACE_DETECTOR_MODEL
)


# ==============================
# FACE LANDMARKER
# ==============================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_LANDMARKER_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    landmarker_options
)
eye_closed_start_time = None
drowsy = False
looking_away_start_time = None
looking_away_alerted = False
looking_away_count = 0
cap = cv2.VideoCapture(0)
interview_start_time = time.time()
total_looking_away_time = 0
yawn_start_time = None
yawn_count = 0
blink_count = 0
eyes_were_closed = False
yawn_was_detected = False
drowsiness_alerted = False
looking_away_duration = 0
attention_status = "CENTER"
cv2.namedWindow("AI Interview Analyzer", cv2.WINDOW_NORMAL)
while True:

    success, frame = cap.read()

    if not success:
        break
    frame = cv2.resize(frame, (1280, 720))

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    
    detection_result = face_detector.detect(mp_image)

    for detection in detection_result.detections:

        score = detection.categories[0].score

        if score < 0.75:
            continue

        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        width = bbox.width
        height = bbox.height

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )



    landmark_result = face_landmarker.detect(mp_image)


    

    if landmark_result.face_landmarks:

        for face_landmarks in landmark_result.face_landmarks:
            nose = face_landmarks[1]
            left_cheek = face_landmarks[234]
            right_cheek = face_landmarks[454]

            face_center_x = (left_cheek.x + right_cheek.x) / 2

            nose_position = nose.x - face_center_x

            if nose_position < -0.05:
                attention_status = "LOOKING LEFT"

            elif nose_position > 0.05:
                attention_status = "LOOKING RIGHT"

            else:
                attention_status = "CENTER"
            if attention_status != "CENTER":

                if looking_away_start_time is None:
                    looking_away_start_time = time.time()

                looking_away_duration = time.time() - looking_away_start_time

            else:

                if looking_away_start_time is not None:
                    total_looking_away_time += time.time() - looking_away_start_time

                looking_away_start_time = None
                looking_away_duration = 0
                looking_away_alerted = False
           

            if looking_away_duration > 2:

                if not looking_away_alerted:
                    winsound.Beep(800, 500)
                    looking_away_alerted = True
                    looking_away_count += 1
            mar = calculate_mar(face_landmarks)
            interview_duration = time.time() - interview_start_time
            #interview_duration = time.time() - interview_start_time
            minutes = int(interview_duration // 60)
            seconds = int(interview_duration % 60)
            if interview_duration > 0:
                attention_percentage = (
                    (interview_duration - total_looking_away_time)
                    / interview_duration
                ) * 100
            else:
                attention_percentage = 100
            attention_percentage = max(0, attention_percentage)
            cv2.putText(
                frame,
                f"Interview Time: {minutes:02d}:{seconds:02d}",
                (30, 520),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Attention Score: {attention_percentage:.1f}%",
                (30, 560),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
            cv2.putText(
            frame,
            f"MAR: {mar:.2f}",
            (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2
            )
            left_eye_indices = [
                33, 160, 158, 133, 153, 144
            ]

            
            right_eye_indices = [
                362, 385, 387, 263, 373, 380
            ]
            left_ear = calculate_ear(
            face_landmarks,
            left_eye_indices
            )

            right_ear = calculate_ear(
            face_landmarks,
            right_eye_indices
            )

            average_ear = (left_ear + right_ear) / 2
           # average_ear = (left_ear + right_ear) / 2

            EAR_THRESHOLD = 0.20

            if average_ear < EAR_THRESHOLD:
                eye_status = "CLOSED"
            else:
                eye_status = "OPEN"
            if eye_status == "CLOSED":
                eyes_were_closed = True

            elif eye_status == "OPEN" and eyes_were_closed:
                blink_count += 1
                eyes_were_closed = False
            if eye_status == "CLOSED":

                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()

                eye_closed_duration = time.time() - eye_closed_start_time

            else:

                eye_closed_start_time = None
                eye_closed_duration = 0

                drowsy = False
                drowsiness_alerted = False


            if eye_closed_duration > 2:

                drowsy = True

                if not drowsiness_alerted:
                    winsound.Beep(1000, 500)
                    drowsiness_alerted = True
            MAR_THRESHOLD = 0.40

            if mar > MAR_THRESHOLD:
                mouth_status = "OPEN"
            else:
                mouth_status = "CLOSED"
            if mouth_status == "OPEN":

                if yawn_start_time is None:
                    yawn_start_time = time.time()

                mouth_open_duration = time.time() - yawn_start_time

            else:
                yawn_start_time = None
                mouth_open_duration = 0
                yawn_was_detected = False


            if mouth_open_duration > 1.2 and not yawn_was_detected:

                yawn_count += 1
                yawn_was_detected = True


            if yawn_was_detected:
                yawn_status = "YAWN DETECTED"
            else:
                yawn_status = "NO YAWN"
            cv2.putText(
            frame,
            f"Eye Aspect Ratio: {average_ear:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
            )
            cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
            )
            cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
            )
            cv2.putText(
            frame,
            yawn_status,
            (30, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
            )
            cv2.putText(
            frame,
            f"Attention: {attention_status}",
            (30, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
            )

            cv2.putText(
            frame,
            f"Away Time: {looking_away_duration:.1f}s",
            (30, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 165, 255),
            2
            )

            cv2.putText(
            frame,
            f"Away Count: {looking_away_count}",
            (30, 480),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 165, 255),
            2
            )
            cv2.putText(
            frame,
            f"Mouth: {mouth_status}",
            (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2


            )
            cv2.putText(
            frame,
            f"Status: {drowsiness_alerted}",
            (30, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
            )
            cv2.putText(
            frame,
            f"Attention: {attention_status}",
            (30, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
            )
            for index in left_eye_indices:

                landmark = face_landmarks[index]

                eye_x = int(
                    landmark.x * frame.shape[1]
                )

                eye_y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (eye_x, eye_y),
                    3,
                    (255, 0, 0),
                    -1
                )


            for index in right_eye_indices:

                landmark = face_landmarks[index]

                eye_x = int(
                    landmark.x * frame.shape[1]
                )

                eye_y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (eye_x, eye_y),
                    3,
                    (255, 0, 0),
                    -1
                )


    # ==============================
    # DISPLAY
    # ==============================

    cv2.namedWindow("AI Interview Analyzer", cv2.WINDOW_NORMAL)

    cv2.imshow(
        "AI Interview Analyzer",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()

face_detector.close()
face_landmarker.close()

cv2.destroyAllWindows()