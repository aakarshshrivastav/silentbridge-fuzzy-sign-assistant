from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silentbridge.config import GESTURE_TYPE_VALUES, QUICK_REPLIES, SIGN_MESSAGES, get_message, messages_table
from silentbridge.fuzzy_engine import FuzzyInputs, SilentBridgeFIS
from silentbridge.gesture_recognition import GestureClassifier
from silentbridge.history import append_history, load_history
from silentbridge.speech import speak_async
from silentbridge.vision import VisionProcessor


st.set_page_config(page_title="SilentBridge", page_icon="SB", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef6f2 45%, #f7efe8 100%);
            color: #17202a;
        }
        [data-testid="stSidebar"] {
            background: #fbfcfd;
            border-right: 1px solid #d9e2e7;
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        .status-panel {
            border: 1px solid #d7e0df;
            border-radius: 8px;
            padding: 16px;
            background: rgba(255, 255, 255, 0.82);
        }
        .emergency {
            border: 2px solid #d73535;
            background: #fff4f1;
            color: #7a1414;
        }
        .message {
            font-size: 1.45rem;
            font-weight: 700;
            margin: 0 0 6px 0;
        }
        .hindi {
            font-size: 1.08rem;
            color: #3e4c59;
        }
        div.stButton > button {
            border-radius: 6px;
            border: 1px solid #9db2aa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def fuzzy_engine() -> SilentBridgeFIS:
    return SilentBridgeFIS()


@st.cache_resource
def classifier() -> GestureClassifier:
    return GestureClassifier(ROOT / "data" / "gesture_centroids.csv")


@st.cache_resource
def vision_processor() -> VisionProcessor:
    return VisionProcessor()


def score_to_text(value: float) -> str:
    return f"{value * 100:.0f}%"


def plot_memberships(engine: SilentBridgeFIS):
    curves = engine.membership_curves()
    fig, axes = plt.subplots(1, len(curves), figsize=(9, 3), constrained_layout=True)
    if len(curves) == 1:
        axes = [axes]
    for axis, (name, sets) in zip(axes, curves.items()):
        for label, y in sets.items():
            axis.plot(engine.universe, y, label=label)
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Score")
        axis.set_ylabel("Membership")
        axis.grid(alpha=0.2)
        axis.legend()
    return fig


def evaluate(label: str, gesture_confidence: float, hand_visibility: float, facial_distress: float, repetition: float, gesture_speed: float):
    message = get_message(label)
    fis_inputs = FuzzyInputs(
        gesture_confidence=gesture_confidence,
        hand_visibility=hand_visibility,
        facial_distress=facial_distress,
        repetition=repetition,
        gesture_speed=gesture_speed,
        gesture_type=GESTURE_TYPE_VALUES[message.gesture_type],
    )
    result = fuzzy_engine().infer(fis_inputs)
    return message, result


def render_result(label: str, gesture_confidence: float, hand_visibility: float, facial_distress: float, repetition: float, gesture_speed: float) -> None:
    message, result = evaluate(label, gesture_confidence, hand_visibility, facial_distress, repetition, gesture_speed)
    is_emergency = result.urgency >= 0.82
    panel_class = "status-panel emergency" if is_emergency else "status-panel"

    st.markdown(
        f"""
        <div class="{panel_class}">
            <div class="message">{message.english}</div>
            <div class="hindi">{message.hindi}</div>
            <p><strong>Detected gesture:</strong> {label}</p>
            <p><strong>Suggested action:</strong> {result.action_level}</p>
            <p><strong>Default response:</strong> {message.default_action}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Gesture Confidence", score_to_text(gesture_confidence))
    cols[1].metric("Communication Score", score_to_text(result.communication_confidence))
    cols[2].metric("Urgency Score", f"{result.urgency:.2f}")
    cols[3].metric("Fuzzy Engine", result.engine)

    if is_emergency:
        st.error("Emergency mode active. Show this screen to nearby people and call for help.")

    action_cols = st.columns([1, 1, 2])
    if action_cols[0].button("Speak Output", use_container_width=True):
        speak_async(message.voice)
    if action_cols[1].button("Log Message", use_container_width=True):
        append_history(
            {
                "gesture": label,
                "message": message.english,
                "communication_confidence": round(result.communication_confidence, 3),
                "urgency": round(result.urgency, 3),
                "action": result.action_level,
            }
        )
        st.toast("Communication logged")


def live_camera_panel() -> None:
    st.subheader("Live Sign Detection")
    start = st.toggle("Start webcam", value=False)
    frame_slot = st.empty()
    status_slot = st.empty()

    if not start:
        status_slot.info("Turn on the webcam toggle to start live detection, or use demo mode from the sidebar.")
        return

    processor = vision_processor()
    model = classifier()
    if not processor.ready:
        status_slot.warning("MediaPipe is not available. Install requirements or use demo mode.")
        return

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        status_slot.error("Could not open webcam. Use demo mode for the presentation.")
        return

    latest = None
    for _ in range(90):
        ok, frame = capture.read()
        if not ok:
            break
        signals = processor.process(frame)
        prediction = model.predict(signals.feature)
        latest = (prediction, signals)
        rgb = cv2.cvtColor(signals.annotated_frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(rgb, channels="RGB", use_container_width=True)
        time.sleep(0.03)

    capture.release()
    if latest is None:
        status_slot.error("No frames were captured from the webcam.")
        return

    prediction, signals = latest
    with status_slot.container():
        render_result(
            prediction.label,
            prediction.confidence,
            signals.hand_visibility,
            signals.facial_distress,
            signals.repetition,
            signals.gesture_speed,
        )


def demo_panel() -> None:
    st.subheader("Demo Mode")
    selected = st.selectbox("Gesture", list(SIGN_MESSAGES.keys()), index=3)
    cols = st.columns(5)
    gesture_confidence = cols[0].slider("Confidence", 0.0, 1.0, 0.87, 0.01)
    hand_visibility = cols[1].slider("Visibility", 0.0, 1.0, 0.82, 0.01)
    facial_distress = cols[2].slider("Distress", 0.0, 1.0, 0.78, 0.01)
    repetition = cols[3].slider("Repetition", 0.0, 1.0, 0.74, 0.01)
    gesture_speed = cols[4].slider("Speed", 0.0, 1.0, 0.68, 0.01)
    render_result(selected, gesture_confidence, hand_visibility, facial_distress, repetition, gesture_speed)


def quick_reply_panel() -> None:
    st.subheader("Two-Way Quick Replies")
    cols = st.columns(len(QUICK_REPLIES))
    for col, reply in zip(cols, QUICK_REPLIES):
        if col.button(reply, use_container_width=True):
            speak_async(reply)
            st.toast(f"Spoken: {reply}")


def main() -> None:
    inject_styles()
    st.title("SilentBridge")
    st.caption("Real-time sign-to-speech and emergency assistance using fuzzy logic")

    with st.sidebar:
        st.header("Control")
        mode = st.radio("Mode", ["Demo", "Live Webcam"], horizontal=False)
        st.divider()
        st.write("Classifier")
        st.write("Trained centroid model found." if classifier().is_trained else "Using demo/fallback classifier.")
        st.write("MediaPipe ready." if vision_processor().ready else "MediaPipe not available.")

    if mode == "Live Webcam":
        live_camera_panel()
    else:
        demo_panel()

    quick_reply_panel()

    tabs = st.tabs(["History", "Sign Catalog", "Fuzzy Memberships"])
    with tabs[0]:
        history = load_history(ROOT / "data" / "communication_history.csv")
        if history.empty:
            st.info("No communication history yet.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(messages_table(), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.pyplot(plot_memberships(fuzzy_engine()), use_container_width=True)


if __name__ == "__main__":
    main()
