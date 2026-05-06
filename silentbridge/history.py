from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


HISTORY_PATH = Path("data/communication_history.csv")


def append_history(row: dict, path: Path = HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    frame = pd.DataFrame([payload])
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def load_history(path: Path = HISTORY_PATH, limit: int = 30) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).tail(limit).iloc[::-1]
