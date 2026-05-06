from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except Exception:  # pragma: no cover - handled at runtime for lightweight demos
    fuzz = None
    ctrl = None


@dataclass(frozen=True)
class FuzzyInputs:
    gesture_confidence: float
    hand_visibility: float
    facial_distress: float
    repetition: float
    gesture_speed: float
    gesture_type: float


@dataclass(frozen=True)
class FuzzyResult:
    communication_confidence: float
    urgency: float
    action_level: str
    engine: str


class SilentBridgeFIS:
    """Mamdani fuzzy inference for communication reliability and urgency."""

    universe = np.arange(0, 1.01, 0.01)

    @cached_property
    def _systems(self):
        if ctrl is None or fuzz is None:
            return None

        gesture_confidence = ctrl.Antecedent(self.universe, "gesture_confidence")
        hand_visibility = ctrl.Antecedent(self.universe, "hand_visibility")
        facial_distress = ctrl.Antecedent(self.universe, "facial_distress")
        repetition = ctrl.Antecedent(self.universe, "repetition")
        gesture_speed = ctrl.Antecedent(self.universe, "gesture_speed")
        gesture_type = ctrl.Antecedent(self.universe, "gesture_type")

        communication_confidence = ctrl.Consequent(self.universe, "communication_confidence", defuzzify_method="centroid")
        urgency = ctrl.Consequent(self.universe, "urgency", defuzzify_method="centroid")

        gesture_confidence["low"] = fuzz.trimf(self.universe, [0.0, 0.0, 0.45])
        gesture_confidence["medium"] = fuzz.trimf(self.universe, [0.25, 0.55, 0.8])
        gesture_confidence["high"] = fuzz.trimf(self.universe, [0.65, 1.0, 1.0])

        hand_visibility["poor"] = fuzz.trimf(self.universe, [0.0, 0.0, 0.45])
        hand_visibility["clear"] = fuzz.trimf(self.universe, [0.25, 0.55, 0.8])
        hand_visibility["excellent"] = fuzz.trimf(self.universe, [0.65, 1.0, 1.0])

        facial_distress["calm"] = fuzz.trimf(self.universe, [0.0, 0.0, 0.4])
        facial_distress["worried"] = fuzz.trimf(self.universe, [0.25, 0.55, 0.8])
        facial_distress["distressed"] = fuzz.trimf(self.universe, [0.65, 1.0, 1.0])

        repetition["low"] = fuzz.trimf(self.universe, [0.0, 0.0, 0.4])
        repetition["medium"] = fuzz.trimf(self.universe, [0.25, 0.55, 0.8])
        repetition["high"] = fuzz.trimf(self.universe, [0.65, 1.0, 1.0])

        gesture_speed["slow"] = fuzz.trimf(self.universe, [0.0, 0.0, 0.42])
        gesture_speed["normal"] = fuzz.trimf(self.universe, [0.25, 0.5, 0.78])
        gesture_speed["fast"] = fuzz.trimf(self.universe, [0.62, 1.0, 1.0])

        gesture_type["normal"] = fuzz.trimf(self.universe, [0.0, 0.1, 0.35])
        gesture_type["need"] = fuzz.trimf(self.universe, [0.35, 0.55, 0.75])
        gesture_type["emergency"] = fuzz.trimf(self.universe, [0.7, 1.0, 1.0])

        communication_confidence["repeat"] = fuzz.trimf(self.universe, [0.0, 0.0, 0.4])
        communication_confidence["medium"] = fuzz.trimf(self.universe, [0.25, 0.55, 0.78])
        communication_confidence["high"] = fuzz.trimf(self.universe, [0.65, 1.0, 1.0])

        urgency["normal"] = fuzz.trimf(self.universe, [0.0, 0.0, 0.35])
        urgency["attention"] = fuzz.trimf(self.universe, [0.25, 0.5, 0.7])
        urgency["high"] = fuzz.trimf(self.universe, [0.6, 0.78, 0.92])
        urgency["critical"] = fuzz.trimf(self.universe, [0.82, 1.0, 1.0])

        confidence_rules = [
            ctrl.Rule(gesture_confidence["high"] & hand_visibility["excellent"], communication_confidence["high"]),
            ctrl.Rule(gesture_confidence["high"] & hand_visibility["clear"], communication_confidence["high"]),
            ctrl.Rule(gesture_confidence["medium"] & repetition["high"], communication_confidence["medium"]),
            ctrl.Rule(gesture_confidence["medium"] & hand_visibility["clear"], communication_confidence["medium"]),
            ctrl.Rule(gesture_confidence["low"] | hand_visibility["poor"], communication_confidence["repeat"]),
            ctrl.Rule(gesture_confidence["low"] & repetition["high"], communication_confidence["medium"]),
        ]

        urgency_rules = [
            ctrl.Rule(gesture_type["emergency"] & facial_distress["distressed"], urgency["critical"]),
            ctrl.Rule(gesture_type["emergency"] & repetition["high"], urgency["critical"]),
            ctrl.Rule(gesture_type["emergency"] & gesture_speed["fast"], urgency["high"]),
            ctrl.Rule(gesture_type["need"] & facial_distress["distressed"], urgency["high"]),
            ctrl.Rule(gesture_type["need"] & repetition["high"], urgency["attention"]),
            ctrl.Rule(gesture_type["normal"] & facial_distress["calm"], urgency["normal"]),
            ctrl.Rule(gesture_type["normal"] & gesture_confidence["high"], urgency["normal"]),
            ctrl.Rule(gesture_confidence["low"] & hand_visibility["poor"], urgency["normal"]),
            ctrl.Rule(gesture_type["need"] & facial_distress["calm"], urgency["attention"]),
        ]

        return (
            ctrl.ControlSystem(confidence_rules),
            ctrl.ControlSystem(urgency_rules),
        )

    def infer(self, inputs: FuzzyInputs) -> FuzzyResult:
        values = {key: self._clip(value) for key, value in inputs.__dict__.items()}
        systems = self._systems
        if systems is None:
            return self._fallback(values)

        confidence_sim = ctrl.ControlSystemSimulation(systems[0])
        urgency_sim = ctrl.ControlSystemSimulation(systems[1])
        for key in ("gesture_confidence", "hand_visibility", "repetition"):
            confidence_sim.input[key] = values[key]
        for key in ("gesture_confidence", "hand_visibility", "facial_distress", "repetition", "gesture_speed", "gesture_type"):
            urgency_sim.input[key] = values[key]

        confidence_sim.compute()
        urgency_sim.compute()

        communication_confidence = float(confidence_sim.output["communication_confidence"])
        urgency = float(urgency_sim.output["urgency"])
        return FuzzyResult(
            communication_confidence=communication_confidence,
            urgency=urgency,
            action_level=self.action_level(communication_confidence, urgency),
            engine="scikit-fuzzy Mamdani",
        )

    def membership_curves(self) -> dict[str, dict[str, np.ndarray]]:
        if fuzz is None:
            x = self.universe
            return {
                "gesture_confidence": {
                    "low": np.maximum(0, 1 - x / 0.45),
                    "medium": np.maximum(0, 1 - np.abs(x - 0.55) / 0.3),
                    "high": np.maximum(0, (x - 0.65) / 0.35),
                }
            }

        x = self.universe
        return {
            "gesture_confidence": {
                "low": fuzz.trimf(x, [0.0, 0.0, 0.45]),
                "medium": fuzz.trimf(x, [0.25, 0.55, 0.8]),
                "high": fuzz.trimf(x, [0.65, 1.0, 1.0]),
            },
            "urgency": {
                "normal": fuzz.trimf(x, [0.0, 0.0, 0.35]),
                "attention": fuzz.trimf(x, [0.25, 0.5, 0.7]),
                "high": fuzz.trimf(x, [0.6, 0.78, 0.92]),
                "critical": fuzz.trimf(x, [0.82, 1.0, 1.0]),
            },
        }

    @staticmethod
    def action_level(communication_confidence: float, urgency: float) -> str:
        if communication_confidence < 0.35:
            return "Ask user to repeat the sign slowly."
        if urgency >= 0.82:
            return "Critical alert: call help immediately."
        if urgency >= 0.65:
            return "High priority: assist the person now."
        if urgency >= 0.42:
            return "Attention needed: respond and confirm."
        return "Normal: reply or acknowledge."

    @staticmethod
    def _clip(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _fallback(self, values: dict[str, float]) -> FuzzyResult:
        communication_confidence = (
            values["gesture_confidence"] * 0.58
            + values["hand_visibility"] * 0.28
            + values["repetition"] * 0.14
        )
        urgency = (
            values["gesture_type"] * 0.42
            + values["facial_distress"] * 0.24
            + values["repetition"] * 0.18
            + values["gesture_speed"] * 0.16
        )
        if values["gesture_confidence"] < 0.35 and values["hand_visibility"] < 0.35:
            communication_confidence *= 0.65
            urgency *= 0.7

        communication_confidence = self._clip(communication_confidence)
        urgency = self._clip(urgency)
        return FuzzyResult(
            communication_confidence=communication_confidence,
            urgency=urgency,
            action_level=self.action_level(communication_confidence, urgency),
            engine="weighted fallback",
        )
