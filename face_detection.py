import cv2
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mediapipe as mp

FACE_DETECTOR_MODEL = "models/blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = "models/face_landmarker.task"
API_HOST = "127.0.0.1"
API_PORT = 5000

state_lock = threading.Lock()
latest_frame = None
state_version = 0
stop_event = threading.Event()
analysis_thread = None
session_summary = None
latest_state = {}


def calculate_ear(landmarks, eye_indices):
    points = [landmarks[index] for index in eye_indices]
    vertical_1 = math.dist((points[1].x, points[1].y), (points[5].x, points[5].y))
    vertical_2 = math.dist((points[2].x, points[2].y), (points[4].x, points[4].y))
    horizontal = math.dist((points[0].x, points[0].y), (points[3].x, points[3].y))
    return (vertical_1 + vertical_2) / (2 * horizontal)


def calculate_mar(landmarks):
    vertical = math.dist((landmarks[13].x, landmarks[13].y), (landmarks[14].x, landmarks[14].y))
    horizontal = math.dist((landmarks[61].x, landmarks[61].y), (landmarks[291].x, landmarks[291].y))
    return vertical / horizontal


def publish(values, frame=None):
    global latest_frame, state_version
    with state_lock:
        latest_state.update(values)
        latest_state["last_updated"] = time.time()
        latest_frame = frame
        state_version += 1


def read_state():
    with state_lock:
        return dict(latest_state)


def read_frame():
    with state_lock:
        return latest_frame


def reset_state():
    global session_summary
    with state_lock:
        session_summary = None
        latest_state.clear()
        latest_state.update({
            "camera": "starting", "face_detected": False, "ear": None, "mar": None,
            "blink_count": 0, "yawn_count": 0, "looking_away_count": 0,
            "eye_status": None, "mouth_status": None, "yawn_status": None,
            "drowsiness_status": None, "attention_status": None,
            "attention_percentage": None, "interview_duration": 0, "last_updated": None,
        })


def finish_session(summary):
    global session_summary
    with state_lock:
        session_summary = summary
    publish({"camera": "stopped", "face_detected": False})


def end_session():
    stop_event.set()
    thread = analysis_thread
    if thread and thread is not threading.current_thread():
        thread.join(timeout=5)
    return read_summary()


def read_summary():
    with state_lock:
        return dict(session_summary) if session_summary else None


class AnalyzerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/analysis":
            self.send_json(read_state())
        elif self.path == "/api/summary":
            self.send_json(read_summary() or {})
        elif self.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            last_version = -1
            try:
                while True:
                    with state_lock:
                        version = state_version
                        state = dict(latest_state)
                    if version != last_version:
                        self.wfile.write(f"data: {json.dumps(state)}\n\n".encode())
                        self.wfile.flush()
                        last_version = version
                    if state.get("camera") == "stopped":
                        break
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    frame = read_frame()
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                        self.wfile.flush()
                    if read_state().get("camera") == "stopped":
                        break
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/end":
            self.send_json(end_session() or {})
        elif self.path == "/api/start":
            start_analysis()
            self.send_json(read_state())
        else:
            self.send_error(404)

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def analysis_loop():
    global latest_state
    detector = mp.tasks.vision.FaceDetector.create_from_model_path(FACE_DETECTOR_MODEL)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    eye_closed_start = None
    yawn_start = None
    eyes_were_closed = False
    yawn_detected = False
    drowsiness_active = False
    closed_eye_time = 0
    drowsiness_events = 0
    looking_away_count = 0
    looking_away_start = None
    looking_away_alerted = False
    total_looking_away_time = 0
    blink_count = 0
    yawn_count = 0
    ear_values = []
    mar_values = []
    max_mar = None
    frame_number = 0
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]

    try:
        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                publish({"camera": "error", "face_detected": False}, None)
                break
            frame_number += 1
            frame = cv2.resize(frame, (1280, 720))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = detector.detect(image)
            for detection in detections.detections:
                if detection.categories[0].score >= 0.75:
                    box = detection.bounding_box
                    cv2.rectangle(frame, (box.origin_x, box.origin_y), (box.origin_x + box.width, box.origin_y + box.height), (0, 255, 0), 2)

            result = landmarker.detect(image)
            duration = time.time() - start_time
            values = {
                "camera": "active", "face_detected": bool(result.face_landmarks), "ear": None, "mar": None,
                "eye_status": None, "mouth_status": None, "yawn_status": None, "drowsiness_status": None,
                "attention_status": None, "looking_away_count": looking_away_count,
                "looking_away_duration": 0, "attention_percentage": None, "interview_duration": duration,
                "blink_count": blink_count, "yawn_count": yawn_count,
            }
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                nose_position = landmarks[1].x - (landmarks[234].x + landmarks[454].x) / 2
                attention_status = "LOOKING LEFT" if nose_position < -0.05 else "LOOKING RIGHT" if nose_position > 0.05 else "CENTER"
                if attention_status != "CENTER":
                    looking_away_start = looking_away_start or time.time()
                    looking_away_duration = time.time() - looking_away_start
                    if looking_away_duration > 2 and not looking_away_alerted:
                        looking_away_count += 1
                        looking_away_alerted = True
                else:
                    if looking_away_start:
                        total_looking_away_time += time.time() - looking_away_start
                    looking_away_start = None
                    looking_away_duration = 0
                    looking_away_alerted = False
                ear = (calculate_ear(landmarks, left_eye) + calculate_ear(landmarks, right_eye)) / 2
                mar = calculate_mar(landmarks)
                ear_values.append(ear)
                mar_values.append(mar)
                max_mar = mar if max_mar is None else max(max_mar, mar)
                eye_status = "CLOSED" if ear < 0.20 else "OPEN"
                mouth_status = "OPEN" if mar > 0.40 else "CLOSED"
                if eye_status == "CLOSED":
                    eyes_were_closed = True
                    eye_closed_start = eye_closed_start or time.time()
                    if time.time() - eye_closed_start > 2 and not drowsiness_active:
                        drowsiness_active = True
                        drowsiness_events += 1
                else:
                    if eye_closed_start:
                        closed_eye_time += time.time() - eye_closed_start
                    eye_closed_start = None
                    if eyes_were_closed:
                        blink_count += 1
                    eyes_were_closed = False
                    drowsiness_active = False
                if mouth_status == "OPEN":
                    yawn_start = yawn_start or time.time()
                else:
                    yawn_start = None
                    yawn_detected = False
                if yawn_start and time.time() - yawn_start > 1.2 and not yawn_detected:
                    yawn_count += 1
                    yawn_detected = True
                values.update({
                    "ear": ear, "mar": mar, "eye_status": eye_status, "mouth_status": mouth_status,
                    "yawn_status": "YAWN DETECTED" if yawn_detected else "NO YAWN",
                    "drowsiness_status": "DROWSY" if drowsiness_active else "ALERT",
                    "attention_status": attention_status, "looking_away_count": looking_away_count,
                    "looking_away_duration": looking_away_duration,
                    "attention_percentage": max(0, ((duration - total_looking_away_time) / duration) * 100),
                    "blink_count": blink_count, "yawn_count": yawn_count,
                })

            ok, encoded = cv2.imencode(".jpg", frame)
            publish(values, encoded.tobytes() if ok else None)
            if frame_number % 30 == 0:
                print(f"Processed frame {frame_number}: EAR={values['ear']} MAR={values['mar']} blinks={blink_count} yawns={yawn_count}")
            cv2.imshow("AI Interview Analyzer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
    finally:
        if eye_closed_start:
            closed_eye_time += time.time() - eye_closed_start
        end_time = time.time()
        duration = end_time - start_time
        summary = {
            "call_start_time": start_time, "call_end_time": end_time, "call_duration": duration,
            "total_blinks": blink_count, "total_yawns": yawn_count,
            "average_ear": sum(ear_values) / len(ear_values) if ear_values else None,
            "average_mar": sum(mar_values) / len(mar_values) if mar_values else None,
            "maximum_mar": max_mar, "drowsiness_events": drowsiness_events,
            "total_eyes_closed_time": closed_eye_time,
            "overall_eye_status": "CLOSED" if eyes_were_closed else "OPEN",
            "overall_drowsiness_status": "DROWSY" if drowsiness_events else "ALERT",
        }
        finish_session(summary)
        cap.release()
        detector.close()
        landmarker.close()
        cv2.destroyAllWindows()


def start_analysis():
    global analysis_thread
    if analysis_thread and analysis_thread.is_alive():
        return
    reset_state()
    stop_event.clear()
    analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
    analysis_thread.start()


def main():
    server = ThreadingHTTPServer((API_HOST, API_PORT), AnalyzerHandler)
    start_analysis()
    print(f"Live analysis API: http://{API_HOST}:{API_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        end_session()
        server.shutdown()


if __name__ == "__main__":
    main()
