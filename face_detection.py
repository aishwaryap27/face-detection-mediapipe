import cv2
import mediapipe as mp

# Path to our face detection model
MODEL_PATH = "models/blaze_face_short_range.tflite"

# Create the face detector using the model
detector = mp.tasks.vision.FaceDetector.create_from_model_path(MODEL_PATH)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if not success:
        break

    # Convert OpenCV's BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame into a MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect faces
    result = detector.detect(mp_image)

    # Draw a box around each detected face
    for detection in result.detections:

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

    # Show the webcam
    cv2.imshow("AI Interview Analyzer", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
cap.release()

# Close the detector
detector.close()

# Close OpenCV windows
cv2.destroyAllWindows()