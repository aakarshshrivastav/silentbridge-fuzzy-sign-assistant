from __future__ import annotations

import threading


def speak_async(text: str) -> None:
    thread = threading.Thread(target=_speak, args=(text,), daemon=True)
    thread.start()


def _speak(text: str) -> None:
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        return
