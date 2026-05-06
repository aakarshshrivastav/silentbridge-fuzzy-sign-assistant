from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silentbridge.gesture_recognition import FEATURE_SIZE
from silentbridge.vision import VisionProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record MediaPipe hand landmark samples for one sign.")
    parser.add_argument("--label", required=True, help="Gesture label, for example Help or Water.")
    parser.add_argument("--samples", type=int, default=80, help="Number of landmark samples to save.")
    parser.add_argument("--output", default="data/gesture_samples.csv", help="CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = VisionProcessor()
    if not processor.ready:
        raise RuntimeError("MediaPipe is not available. Install requirements.txt first.")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    rows = []
    while len(rows) < args.samples:
        ok, frame = capture.read()
        if not ok:
            break
        signals = processor.process(frame)
        display = signals.annotated_frame.copy()
        cv2.putText(display, f"{args.label}: {len(rows)}/{args.samples}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 180, 60), 2)
        cv2.imshow("SilentBridge Recorder", display)
        if signals.feature is not None:
            rows.append({"label": args.label, **{f"f{i}": signals.feature[i] for i in range(FEATURE_SIZE)}})
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
    processor.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(output, mode="a", index=False, header=not output.exists())
    print(f"Saved {len(rows)} samples to {output}")


if __name__ == "__main__":
    main()
