from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from silentbridge.gesture_recognition import landmarks_to_feature

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional runtime dependency
    mp = None


@dataclass(frozen=True)
class FrameSignals:
    annotated_frame: np.ndarray
    feature: np.ndarray | None
    hand_visibility: float
    facial_distress: float
    repetition: float
    gesture_speed: float
    hand_detected: bool


class VisionProcessor:
    def __init__(self):
        self.previous_center: np.ndarray | None = None
        self.previous_feature: np.ndarray | None = None
        self.repetition_memory: list[float] = []
        self.hands: Any | None = None
        self.face_mesh: Any | None = None
        self.drawer: Any | None = None
        self._init_mediapipe()

    @property
    def ready(self) -> bool:
        return mp is not None and self.hands is not None

    def process(self, frame_bgr: np.ndarray) -> FrameSignals:
        if not self.ready:
            return FrameSignals(frame_bgr, None, 0.2, 0.1, 0.1, 0.1, False)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb)
        face_results = self.face_mesh.process(rgb) if self.face_mesh else None

        annotated = frame_bgr.copy()
        feature = None
        hand_visibility = 0.18
        hand_detected = False
        speed = 0.08
        repetition = self._repetition_score(None)

        if hand_results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            feature = landmarks_to_feature(hand_landmarks)
            points = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            center = points.mean(axis=0)
            visible_points = np.mean((points[:, 0] >= 0) & (points[:, 0] <= 1) & (points[:, 1] >= 0) & (points[:, 1] <= 1))
            hand_visibility = float(np.clip(0.25 + visible_points * 0.75, 0.0, 1.0))
            speed = self._speed_score(center)
            repetition = self._repetition_score(feature)
            self.drawer.draw_landmarks(annotated, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        distress = self._distress_score(face_results)
        return FrameSignals(annotated, feature, hand_visibility, distress, repetition, speed, hand_detected)

    def close(self) -> None:
        if self.hands:
            self.hands.close()
        if self.face_mesh:
            self.face_mesh.close()

    def _init_mediapipe(self) -> None:
        if mp is None:
            return
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawer = mp.solutions.drawing_utils

    def _speed_score(self, center: np.ndarray) -> float:
        if self.previous_center is None:
            self.previous_center = center
            return 0.15
        delta = float(np.linalg.norm(center - self.previous_center))
        self.previous_center = center
        return float(np.clip(delta * 12.0, 0.0, 1.0))

    def _repetition_score(self, feature: np.ndarray | None) -> float:
        if feature is None:
            return 0.1
        if self.previous_feature is None:
            self.previous_feature = feature
            return 0.2
        similarity = float(np.exp(-np.linalg.norm(feature - self.previous_feature)))
        self.previous_feature = feature
        self.repetition_memory.append(similarity)
        self.repetition_memory = self.repetition_memory[-24:]
        return float(np.clip(np.mean(self.repetition_memory), 0.0, 1.0))

    def _distress_score(self, face_results) -> float:
        if not face_results or not face_results.multi_face_landmarks:
            return 0.18
        landmarks = face_results.multi_face_landmarks[0].landmark
        try:
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            left_brow = landmarks[70]
            right_brow = landmarks[300]
            mouth_open = abs(lower_lip.y - upper_lip.y)
            brow_tilt = abs(left_brow.y - right_brow.y)
            return float(np.clip((mouth_open * 12.0) + (brow_tilt * 8.0), 0.05, 1.0))
        except Exception:
            return 0.25
