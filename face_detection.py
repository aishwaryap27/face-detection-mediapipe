import cv2
import mediapipe as mp
import math
import time


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/face_landmarker.task"

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.40

DROWSINESS_TIME = 2.0
YAWN_TIME = 1.2
LOOKING_AWAY_TIME = 2.0


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# ============================================================
# EYE LANDMARKS
# ============================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ============================================================
# CALCULATE EAR
# ============================================================

def calculate_ear(landmarks, eye_indices):

    points = [
        landmarks[index]
        for index in eye_indices
    ]

    vertical_1 = math.dist(
        (points[1].x, points[1].y),
        (points[5].x, points[5].y)
    )

    vertical_2 = math.dist(
        (points[2].x, points[2].y),
        (points[4].x, points[4].y)
    )

    horizontal = math.dist(
        (points[0].x, points[0].y),
        (points[3].x, points[3].y)
    )

    if horizontal == 0:
        return 0

    ear = (
        vertical_1 + vertical_2
    ) / (2 * horizontal)

    return ear


# ============================================================
# CALCULATE MAR
# ============================================================

def calculate_mar(landmarks):

    vertical = math.dist(
        (landmarks[13].x, landmarks[13].y),
        (landmarks[14].x, landmarks[14].y)
    )

    horizontal = math.dist(
        (landmarks[61].x, landmarks[61].y),
        (landmarks[291].x, landmarks[291].y)
    )

    if horizontal == 0:
        return 0

    mar = vertical / horizontal

    return mar


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(frame, text, x, y):

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(seconds):

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 50)
    print("AI INTERVIEW ANALYZER")
    print("=" * 50)

    print("Starting camera...")
    print("Press Q to quit.")

    # --------------------------------------------------------
    # MEDIAPIPE FACE LANDMARKER
    # --------------------------------------------------------

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.IMAGE,
        num_faces=1
    )

    landmarker = FaceLandmarker.create_from_options(
        options
    )

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")

        landmarker.close()

        return

    # --------------------------------------------------------
    # SESSION VARIABLES
    # --------------------------------------------------------

    start_time = time.time()

    eye_closed_start = None
    yawn_start = None
    looking_away_start = None

    eyes_were_closed = False
    yawn_counted = False

    blink_count = 0
    yawn_count = 0
    drowsiness_count = 0
    looking_away_count = 0

    drowsiness_active = False

    total_looking_away_time = 0

    ear_values = []
    mar_values = []

    # --------------------------------------------------------
    # CAMERA LOOP
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:

            print("Could not read camera.")

            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Resize
        frame = cv2.resize(
            frame,
            (1280, 720)
        )

        # ----------------------------------------------------
        # CONVERT BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------------------
        # FACE LANDMARK DETECTION
        # ----------------------------------------------------

        result = landmarker.detect(image)

        # ----------------------------------------------------
        # DEFAULT VALUES
        # ----------------------------------------------------

        face_detected = False

        ear = None
        mar = None

        eye_status = "NO FACE"
        mouth_status = "NO FACE"

        attention_status = "NO FACE"
        drowsiness_status = "NO FACE"
        yawn_status = "NO FACE"

        # ----------------------------------------------------
        # FACE FOUND
        # ----------------------------------------------------

        if result.face_landmarks:

            face_detected = True

            landmarks = result.face_landmarks[0]

            # =================================================
            # EAR
            # =================================================

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE
            )

            ear = (
                left_ear + right_ear
            ) / 2

            ear_values.append(ear)

            # -------------------------------------------------
            # EYE STATUS
            # -------------------------------------------------

            if ear < EAR_THRESHOLD:

                eye_status = "CLOSED"

            else:

                eye_status = "OPEN"

            # =================================================
            # MAR
            # =================================================

            mar = calculate_mar(landmarks)

            mar_values.append(mar)

            # -------------------------------------------------
            # MOUTH STATUS
            # -------------------------------------------------

            if mar > MAR_THRESHOLD:

                mouth_status = "OPEN"

            else:

                mouth_status = "CLOSED"

            # =================================================
            # BLINK + DROWSINESS
            # =================================================

            if eye_status == "CLOSED":

                if eye_closed_start is None:

                    eye_closed_start = time.time()

                eyes_were_closed = True

                closed_duration = (
                    time.time() -
                    eye_closed_start
                )

                # DROWSINESS

                if (
                    closed_duration > DROWSINESS_TIME
                    and not drowsiness_active
                ):

                    drowsiness_active = True

                    drowsiness_count += 1

            else:

                # Eyes opened again

                if eyes_were_closed:

                    blink_count += 1

                eyes_were_closed = False

                eye_closed_start = None

                drowsiness_active = False

            # =================================================
            # YAWN DETECTION
            # =================================================

            if mouth_status == "OPEN":

                if yawn_start is None:

                    yawn_start = time.time()

                yawn_duration = (
                    time.time() -
                    yawn_start
                )

                if (
                    yawn_duration > YAWN_TIME
                    and not yawn_counted
                ):

                    yawn_count += 1

                    yawn_counted = True

            else:

                yawn_start = None

                yawn_counted = False

            if yawn_counted:

                yawn_status = "YAWN"

            else:

                yawn_status = "NO YAWN"

            # =================================================
            # ATTENTION / HEAD DIRECTION
            # =================================================

            nose_x = landmarks[1].x

            left_face_x = landmarks[234].x
            right_face_x = landmarks[454].x

            face_center = (
                left_face_x +
                right_face_x
            ) / 2

            difference = (
                nose_x -
                face_center
            )

            if difference < -0.05:

                attention_status = "LOOKING LEFT"

            elif difference > 0.05:

                attention_status = "LOOKING RIGHT"

            else:

                attention_status = "CENTER"

            # =================================================
            # LOOKING AWAY TIMER
            # =================================================

            if attention_status != "CENTER":

                if looking_away_start is None:

                    looking_away_start = time.time()

                away_duration = (
                    time.time() -
                    looking_away_start
                )

                if (
                    away_duration > LOOKING_AWAY_TIME
                ):

                    if looking_away_count == 0:

                        looking_away_count += 1

            else:

                if looking_away_start is not None:

                    total_looking_away_time += (
                        time.time() -
                        looking_away_start
                    )

                looking_away_start = None

                # Reset so another looking-away
                # event can be counted

                if looking_away_count > 0:

                    pass

            # =================================================
            # DRAW FACE LANDMARKS
            # =================================================

            # Left eye

            for index in LEFT_EYE:

                point = landmarks[index]

                x = int(
                    point.x *
                    frame.shape[1]
                )

                y = int(
                    point.y *
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (255, 0, 0),
                    -1
                )

            # Right eye

            for index in RIGHT_EYE:

                point = landmarks[index]

                x = int(
                    point.x *
                    frame.shape[1]
                )

                y = int(
                    point.y *
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (255, 0, 0),
                    -1
                )

            # Mouth

            mouth_points = [
                13,
                14,
                61,
                291
            ]

            for index in mouth_points:

                point = landmarks[index]

                x = int(
                    point.x *
                    frame.shape[1]
                )

                y = int(
                    point.y *
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1
                )

            # =================================================
            # DROWSINESS STATUS
            # =================================================

            if drowsiness_active:

                drowsiness_status = "DROWSY"

            else:

                drowsiness_status = "ALERT"

        # ====================================================
        # SESSION TIME
        # ====================================================

        duration = (
            time.time() -
            start_time
        )

        # ====================================================
        # ATTENTION PERCENTAGE
        # ====================================================

        current_away_time = 0

        if looking_away_start is not None:

            current_away_time = (
                time.time() -
                looking_away_start
            )

        total_away = (
            total_looking_away_time +
            current_away_time
        )

        if duration > 0:

            attention_percentage = (
                (duration - total_away)
                / duration
            ) * 100

            attention_percentage = max(
                0,
                min(100, attention_percentage)
            )

        else:

            attention_percentage = 100

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (15, 15),
            (500, 460),
            (0, 0, 0),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.65,
            frame,
            0.35,
            0
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        draw_text(
            frame,
            "AI INTERVIEW ANALYZER",
            30,
            50
        )

        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        face_text = (
            "DETECTED"
            if face_detected
            else
            "NOT DETECTED"
        )

        draw_text(
            frame,
            "Face: " + face_text,
            30,
            85
        )

        # ----------------------------------------------------
        # EYES
        # ----------------------------------------------------

        if ear is not None:

            draw_text(
                frame,
                f"EAR: {ear:.3f}",
                30,
                125
            )

            draw_text(
                frame,
                "Eyes: " + eye_status,
                30,
                155
            )

        else:

            draw_text(
                frame,
                "EAR: --",
                30,
                125
            )

            draw_text(
                frame,
                "Eyes: --",
                30,
                155
            )

        draw_text(
            frame,
            f"Blinks: {blink_count}",
            30,
            190
        )

        # ----------------------------------------------------
        # MOUTH
        # ----------------------------------------------------

        if mar is not None:

            draw_text(
                frame,
                f"MAR: {mar:.3f}",
                30,
                230
            )

            draw_text(
                frame,
                "Mouth: " + mouth_status,
                30,
                260
            )

        else:

            draw_text(
                frame,
                "MAR: --",
                30,
                230
            )

        draw_text(
            frame,
            f"Yawns: {yawn_count}",
            30,
            295
        )

        draw_text(
            frame,
            "Yawn: " + yawn_status,
            30,
            325
        )

        # ----------------------------------------------------
        # DROWSINESS
        # ----------------------------------------------------

        draw_text(
            frame,
            "Drowsiness: " +
            drowsiness_status,
            30,
            365
        )

        # ----------------------------------------------------
        # ATTENTION
        # ----------------------------------------------------

        draw_text(
            frame,
            "Attention: " +
            attention_status,
            30,
            405
        )

        draw_text(
            frame,
            f"Looking Away: "
            f"{looking_away_count}",
            30,
            440
        )

        # ====================================================
        # RIGHT SIDE
        # ====================================================

        draw_text(
            frame,
            f"Session: "
            f"{format_time(duration)}",
            530,
            50
        )

        draw_text(
            frame,
            f"Attention: "
            f"{attention_percentage:.1f}%",
            530,
            90
        )

        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "AI Interview Analyzer",
            frame
        )

        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    end_time = time.time()

    duration = (
        end_time -
        start_time
    )

    print()
    print("=" * 50)
    print("INTERVIEW SUMMARY")
    print("=" * 50)

    print(
        f"Duration          : "
        f"{format_time(duration)}"
    )

    print(
        f"Blink Count       : "
        f"{blink_count}"
    )

    print(
        f"Yawn Count        : "
        f"{yawn_count}"
    )

    print(
        f"Drowsiness Events : "
        f"{drowsiness_count}"
    )

    print(
        f"Looking Away      : "
        f"{looking_away_count}"
    )

    # --------------------------------------------------------
    # AVERAGE EAR
    # --------------------------------------------------------

    if ear_values:

        average_ear = (
            sum(ear_values) /
            len(ear_values)
        )

        print(
            f"Average EAR       : "
            f"{average_ear:.3f}"
        )

    else:

        print("Average EAR       : --")

    # --------------------------------------------------------
    # AVERAGE MAR
    # --------------------------------------------------------

    if mar_values:

        average_mar = (
            sum(mar_values) /
            len(mar_values)
        )

        print(
            f"Average MAR       : "
            f"{average_mar:.3f}"
        )

    else:

        print("Average MAR       : --")

    print(
        f"Attention         : "
        f"{attention_percentage:.1f}%"
    )

    if drowsiness_count > 0:

        print(
            "Overall Status    : "
            "DROWSINESS DETECTED"
        )

    else:

        print(
            "Overall Status    : "
            "ALERT"
        )

    print("=" * 50)

    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()