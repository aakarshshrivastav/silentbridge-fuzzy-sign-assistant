from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from silentbridge.config import SIGN_MESSAGES


LANDMARK_COUNT = 21
FEATURE_SIZE = LANDMARK_COUNT * 3


@dataclass(frozen=True)
class GesturePrediction:
    label: str
    confidence: float
    distance: float | None = None


def landmarks_to_feature(hand_landmarks) -> np.ndarray:
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=float)
    wrist = points[0].copy()
    points -= wrist
    scale = np.linalg.norm(points[:, :2].max(axis=0) - points[:, :2].min(axis=0))
    if scale > 1e-6:
        points /= scale
    return points.reshape(-1)


class GestureClassifier:
    def __init__(self, centroid_path: Path | str = "data/gesture_centroids.csv"):
        self.centroid_path = Path(centroid_path)
        self.centroids = self._load_centroids()

    @property
    def is_trained(self) -> bool:
        return not self.centroids.empty

    def predict(self, feature: np.ndarray | None, demo_label: str | None = None, demo_confidence: float = 0.87) -> GesturePrediction:
        if demo_label:
            return GesturePrediction(label=demo_label, confidence=float(np.clip(demo_confidence, 0.0, 1.0)))

        if feature is None or not self.is_trained:
            return GesturePrediction(label="I need assistance", confidence=0.42)

        feature = np.asarray(feature, dtype=float)
        feature_columns = [f"f{i}" for i in range(FEATURE_SIZE)]
        matrix = self.centroids[feature_columns].to_numpy(dtype=float)
        distances = np.linalg.norm(matrix - feature, axis=1)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        confidence = float(np.exp(-best_distance * 2.8))
        label = str(self.centroids.iloc[best_index]["label"])
        if label not in SIGN_MESSAGES:
            label = "I need assistance"
        return GesturePrediction(label=label, confidence=np.clip(confidence, 0.05, 0.99), distance=best_distance)

    def _load_centroids(self) -> pd.DataFrame:
        if not self.centroid_path.exists():
            return pd.DataFrame()
        data = pd.read_csv(self.centroid_path)
        expected = {"label", *{f"f{i}" for i in range(FEATURE_SIZE)}}
        if not expected.issubset(data.columns):
            return pd.DataFrame()
        return data
