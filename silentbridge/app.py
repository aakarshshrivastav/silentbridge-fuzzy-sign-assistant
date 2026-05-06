from __future__ import annotations

import base64
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silentbridge.config import GESTURE_TYPE_VALUES, QUICK_REPLIES, SIGN_MESSAGES, get_message, messages_table
from silentbridge.fuzzy_engine import FuzzyInputs, SilentBridgeFIS
from silentbridge.gesture_recognition import GestureClassifier
from silentbridge.history import append_history, load_history
from silentbridge.speech import speak_async
from silentbridge.vision import VisionProcessor

app = Flask(__name__, static_folder=str(ROOT / "silentbridge" / "static"), static_url_path="/static")
fuzzy_engine = SilentBridgeFIS()
classifier = GestureClassifier(ROOT / "data" / "gesture_centroids.csv")


def evaluate_sign(label: str, gesture_confidence: float, hand_visibility: float, facial_distress: float, repetition: float, gesture_speed: float) -> dict[str, Any]:
    message = get_message(label)
    fis_inputs = FuzzyInputs(
        gesture_confidence=gesture_confidence,
        hand_visibility=hand_visibility,
        facial_distress=facial_distress,
        repetition=repetition,
        gesture_speed=gesture_speed,
        gesture_type=GESTURE_TYPE_VALUES[message.gesture_type],
    )
    result = fuzzy_engine.infer(fis_inputs)
    return {
        "label": label,
        "message": asdict(message),
        "inputs": asdict(fis_inputs),
        "communication_confidence": round(result.communication_confidence, 4),
        "urgency": round(result.urgency, 4),
        "action_level": result.action_level,
        "engine": result.engine,
        "emergency": result.urgency >= 0.82,
    }


class CameraWorker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.processor: VisionProcessor | None = None
        self.capture: cv2.VideoCapture | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_result: dict[str, Any] | None = None
        self.error: str | None = None

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return True
            self.processor = VisionProcessor()
            if not self.processor.ready:
                self.error = "MediaPipe classic solutions are not available. Use demo mode or install the pinned requirements."
                return False
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.error = "Could not open webcam. Check camera permission or use demo mode."
                self.capture.release()
                self.capture = None
                return False
            self.running = True
            self.error = None
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            return True

    def stop(self) -> None:
        with self.lock:
            self.running = False
        if self.thread:
            self.thread.join(timeout=1.5)
        with self.lock:
            if self.capture:
                self.capture.release()
            if self.processor:
                self.processor.close()
            self.capture = None
            self.processor = None
            self.thread = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "has_frame": self.latest_jpeg is not None,
                "error": self.error,
                "latest": self.latest_result,
            }

    def frame(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg

    def _loop(self) -> None:
        while True:
            with self.lock:
                if not self.running or self.capture is None or self.processor is None:
                    break
                capture = self.capture
                processor = self.processor

            ok, frame = capture.read()
            if not ok:
                with self.lock:
                    self.error = "Webcam stopped returning frames."
                    self.running = False
                break

            signals = processor.process(frame)
            prediction = classifier.predict(signals.feature)
            result = evaluate_sign(
                prediction.label,
                prediction.confidence,
                signals.hand_visibility,
                signals.facial_distress,
                signals.repetition,
                signals.gesture_speed,
            )
            result["hand_detected"] = signals.hand_detected
            result["gesture_distance"] = prediction.distance

            label = f"{prediction.label} | urgency {result['urgency']:.2f}"
            cv2.putText(signals.annotated_frame, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (34, 139, 230), 2)
            ok, encoded = cv2.imencode(".jpg", signals.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                with self.lock:
                    self.latest_jpeg = encoded.tobytes()
                    self.latest_result = result
            time.sleep(0.03)


camera = CameraWorker()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    return jsonify(
        {
            "classifier_trained": classifier.is_trained,
            "camera": camera.status(),
            "signs": list(SIGN_MESSAGES.keys()),
            "quick_replies": QUICK_REPLIES,
        }
    )


@app.post("/api/evaluate")
def evaluate():
    payload = request.get_json(force=True)
    result = evaluate_sign(
        payload.get("label", "Help"),
        float(payload.get("gesture_confidence", 0.87)),
        float(payload.get("hand_visibility", 0.82)),
        float(payload.get("facial_distress", 0.78)),
        float(payload.get("repetition", 0.74)),
        float(payload.get("gesture_speed", 0.68)),
    )
    return jsonify(result)


@app.post("/api/speak")
def speak():
    payload = request.get_json(force=True)
    text = str(payload.get("text", "")).strip()
    if text:
        speak_async(text)
    return jsonify({"ok": bool(text)})


@app.post("/api/log")
def log_message():
    payload = request.get_json(force=True)
    append_history(
        {
            "gesture": payload.get("label", ""),
            "message": payload.get("message", ""),
            "communication_confidence": payload.get("communication_confidence", ""),
            "urgency": payload.get("urgency", ""),
            "action": payload.get("action_level", ""),
        },
        ROOT / "data" / "communication_history.csv",
    )
    return jsonify({"ok": True})


@app.get("/api/history")
def history():
    data = load_history(ROOT / "data" / "communication_history.csv")
    return jsonify(data.to_dict(orient="records"))


@app.get("/api/catalog")
def catalog():
    return jsonify(messages_table().to_dict(orient="records"))


@app.get("/api/memberships")
def memberships():
    curves = fuzzy_engine.membership_curves()
    return jsonify(
        {
            "x": [round(float(value), 2) for value in fuzzy_engine.universe],
            "curves": {
                group: {name: [round(float(point), 4) for point in values] for name, values in sets.items()}
                for group, sets in curves.items()
            },
        }
    )


@app.post("/api/live/start")
def live_start():
    ok = camera.start()
    return jsonify({"ok": ok, **camera.status()})


@app.post("/api/live/stop")
def live_stop():
    camera.stop()
    return jsonify({"ok": True, **camera.status()})


@app.get("/api/live/latest")
def live_latest():
    return jsonify(camera.status())


@app.get("/api/live/frame")
def live_frame():
    frame = camera.frame()
    if frame is None:
        return jsonify({"error": "No camera frame available."}), 404
    encoded = base64.b64encode(frame).decode("ascii")
    return jsonify({"image": f"data:image/jpeg;base64,{encoded}", "latest": camera.status()["latest"]})


@app.get("/video_feed")
def video_feed():
    def stream():
        while True:
            frame = camera.frame()
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.05)

    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
