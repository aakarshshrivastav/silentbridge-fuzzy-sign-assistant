from __future__ import annotations

import base64
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silentbridge.config import GESTURE_TYPE_VALUES, QUICK_REPLIES, SIGN_MESSAGES, get_message, messages_table
from silentbridge.fuzzy_engine import FuzzyInputs, SilentBridgeFIS
from silentbridge.gesture_recognition import GestureClassifier
from silentbridge.history import append_history, load_history
from silentbridge.speech import speak_async
from silentbridge.vision import VisionProcessor

fuzzy_engine = SilentBridgeFIS()
classifier = GestureClassifier(ROOT / "data" / "gesture_centroids.csv")
vision_processor: VisionProcessor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if vision_processor is not None:
        vision_processor.close()


app = FastAPI(title="SilentBridge", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "silentbridge" / "static"), name="static")


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


def get_vision_processor() -> VisionProcessor:
    global vision_processor
    if vision_processor is None:
        vision_processor = VisionProcessor()
    return vision_processor


@app.get("/")
def index():
    return FileResponse(ROOT / "silentbridge" / "static" / "index.html")


@app.get("/api/status")
def status():
    processor = get_vision_processor()
    return {
        "classifier_trained": classifier.is_trained,
        "camera": {
            "running": False,
            "error": None if processor.ready else "MediaPipe is unavailable. Browser preview still works.",
        },
        "vision_ready": processor.ready,
        "signs": list(SIGN_MESSAGES.keys()),
        "quick_replies": QUICK_REPLIES,
    }


@app.post("/api/evaluate")
async def evaluate(request: Request):
    payload = await request.json()
    return evaluate_sign(
        payload.get("label", "Help"),
        float(payload.get("gesture_confidence", 0.87)),
        float(payload.get("hand_visibility", 0.82)),
        float(payload.get("facial_distress", 0.78)),
        float(payload.get("repetition", 0.74)),
        float(payload.get("gesture_speed", 0.68)),
    )


@app.post("/api/analyze-frame")
async def analyze_frame(request: Request):
    payload = await request.json()
    image_data = str(payload.get("image", ""))
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    if not image_data:
        return {"ok": False, "error": "No image received."}

    processor = get_vision_processor()
    if not processor.ready:
        return {"ok": False, "error": "MediaPipe is unavailable. Browser camera preview is still active."}

    try:
        raw = base64.b64decode(image_data)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return {"ok": False, "error": "Could not decode camera frame."}
    if frame is None:
        return {"ok": False, "error": "Could not read camera frame."}

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
    result["ok"] = True
    result["hand_detected"] = signals.hand_detected
    result["gesture_distance"] = prediction.distance
    return result


@app.post("/api/speak")
async def speak(request: Request):
    payload = await request.json()
    text = str(payload.get("text", "")).strip()
    if text:
        speak_async(text)
    return {"ok": bool(text)}


@app.post("/api/log")
async def log_message(request: Request):
    payload = await request.json()
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
    return {"ok": True}


@app.get("/api/history")
def history():
    data = load_history(ROOT / "data" / "communication_history.csv")
    return data.to_dict(orient="records")


@app.get("/api/catalog")
def catalog():
    return messages_table().to_dict(orient="records")


@app.get("/api/memberships")
def memberships():
    curves = fuzzy_engine.membership_curves()
    return {
        "x": [round(float(value), 2) for value in fuzzy_engine.universe],
        "curves": {
            group: {name: [round(float(point), 4) for point in values] for name, values in sets.items()}
            for group, sets in curves.items()
        },
    }


if __name__ == "__main__":
    uvicorn.run("silentbridge.app:app", host="127.0.0.1", port=5000, reload=False)
