import cv2
import mediapipe as mp


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


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)


while True:

    success, frame = cap.read()

    if not success:
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ==============================
    # FACE DETECTION
    # ==============================

    detection_result = face_detector.detect(mp_image)

    for detection in detection_result.detections:

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


    # ==============================
    # FACE LANDMARKS
    # ==============================

    landmark_result = face_landmarker.detect(mp_image)


    # ==============================
    # EYE LANDMARKS
    # ==============================

    if landmark_result.face_landmarks:

        for face_landmarks in landmark_result.face_landmarks:

            # Left eye
            left_eye_indices = [
                33, 160, 158, 133, 153, 144
            ]

            # Right eye
            right_eye_indices = [
                362, 385, 387, 263, 373, 380
            ]


            # Draw left eye landmarks
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


            # Draw right eye landmarks
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