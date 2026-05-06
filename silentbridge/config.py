from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignMessage:
    label: str
    gesture_type: str
    english: str
    hindi: str
    voice: str
    default_action: str


SIGN_MESSAGES: dict[str, SignMessage] = {
    "Hello": SignMessage("Hello", "normal", "Hello.", "नमस्ते।", "Hello.", "Reply politely."),
    "Yes": SignMessage("Yes", "normal", "Yes.", "हाँ।", "Yes.", "Continue conversation."),
    "No": SignMessage("No", "normal", "No.", "नहीं।", "No.", "Ask a clarifying question."),
    "Help": SignMessage("Help", "emergency", "I need help.", "मुझे मदद चाहिए।", "I need help urgently.", "Alert nearby people."),
    "Pain": SignMessage("Pain", "emergency", "I am in pain.", "मुझे दर्द हो रहा है।", "I am in pain. Please help.", "Ask where the pain is and call help if needed."),
    "Water": SignMessage("Water", "need", "I need water.", "मुझे पानी चाहिए।", "I need water.", "Offer water."),
    "Food": SignMessage("Food", "need", "I need food.", "मुझे खाना चाहिए।", "I need food.", "Offer food or ask dietary preference."),
    "Medicine": SignMessage("Medicine", "emergency", "I need medicine.", "मुझे दवा चाहिए।", "I need medicine urgently.", "Find medicine or call a caregiver."),
    "Emergency": SignMessage("Emergency", "emergency", "This is an emergency.", "यह आपातकाल है।", "Emergency detected. Please call for help.", "Call emergency help."),
    "Thank you": SignMessage("Thank you", "normal", "Thank you.", "धन्यवाद।", "Thank you.", "Acknowledge the message."),
    "I need assistance": SignMessage(
        "I need assistance",
        "need",
        "I need assistance.",
        "मुझे सहायता चाहिए।",
        "I need assistance.",
        "Ask what kind of assistance is needed.",
    ),
}

GESTURE_TYPE_VALUES = {"normal": 0.15, "need": 0.55, "emergency": 0.95}

QUICK_REPLIES = [
    "Are you okay?",
    "Do you need water?",
    "Should I call someone?",
    "Where is the pain?",
    "Do you need medicine?",
]


def messages_table() -> pd.DataFrame:
    return pd.DataFrame([message.__dict__ for message in SIGN_MESSAGES.values()])


def get_message(label: str) -> SignMessage:
    return SIGN_MESSAGES.get(label, SIGN_MESSAGES["I need assistance"])
