# SilentBridge

Real-time sign-to-speech and emergency assistance system for deaf and mute users, using hand landmarks, emotion cues, and Mamdani fuzzy inference.

SilentBridge turns webcam gestures into text, speech, urgency scores, and suggested actions. It includes a live camera pipeline using OpenCV and MediaPipe, a fuzzy inference backend using scikit-fuzzy, and a Streamlit dashboard for demonstration.

## Features

- Live webcam capture with OpenCV
- Hand landmark detection with MediaPipe Hands
- Optional facial distress estimation with MediaPipe Face Mesh
- Gesture classification from recorded landmark centroids
- Demo mode for classroom presentation without a trained dataset
- Mamdani fuzzy inference for communication confidence and urgency
- Centroid defuzzification
- Text output, Hindi/English message display, and text-to-speech
- Emergency mode for Help, Pain, Medicine, and Emergency signs
- Two-way quick replies for non-sign-language users
- Communication history logging
- Membership function visualizations

## Supported Demo Signs

- Hello
- Yes
- No
- Help
- Pain
- Water
- Food
- Medicine
- Emergency
- Thank you
- I need assistance

## Project Structure

```text
silentbridge/
  app.py                    Streamlit frontend and orchestration
  config.py                 Sign catalog, messages, and actions
  fuzzy_engine.py           Mamdani fuzzy inference system
  gesture_recognition.py    Landmark feature extraction and classifier
  history.py                CSV communication logging
  speech.py                 Text-to-speech helpers
  vision.py                 Webcam frame processing
scripts/
  record_gesture_dataset.py Webcam dataset recorder
  train_centroid_model.py   Builds a centroid model from recorded landmarks
data/
  gesture_samples.csv       Created by recorder
  gesture_centroids.csv     Created by trainer
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run silentbridge/app.py
```

If you do not have a webcam, use the dashboard's demo mode.

## Train Custom Gestures

Record samples for each sign:

```bash
python3 scripts/record_gesture_dataset.py --label Help --samples 80
python3 scripts/record_gesture_dataset.py --label Water --samples 80
python3 scripts/record_gesture_dataset.py --label Emergency --samples 80
```

Train the simple centroid classifier:

```bash
python3 scripts/train_centroid_model.py
```

Then restart the Streamlit app. The live classifier will load `data/gesture_centroids.csv` automatically.

## Fuzzy Inputs

| Input | Fuzzy Sets |
| --- | --- |
| Gesture confidence | low, medium, high |
| Hand visibility | poor, clear, excellent |
| Facial distress | calm, worried, distressed |
| Gesture repetition | low, medium, high |
| Gesture speed | slow, normal, fast |
| Gesture type | normal, need-based, emergency |

## Fuzzy Outputs

| Output | Meaning |
| --- | --- |
| Communication confidence | Reliability of the interpreted sign |
| Urgency score | How urgent the message/action is |
| Suggested action | Repeat, reply, assist, alert, or call help |

## Notes

This is a college PBL-friendly implementation. It works as a complete demo immediately, and it can become more accurate by recording more landmark samples or replacing the centroid classifier with a trained ML model.
