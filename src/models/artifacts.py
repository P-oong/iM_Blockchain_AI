"""UTF-8 reports in both CSV and strict JSON; no NaN/Infinity JSON literals."""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def clean(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class Reports:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def table(self, name, frame):
        frame = pd.DataFrame(frame)
        frame.to_csv(self.directory / f"{name}.csv", index=False, encoding="utf-8-sig")
        # Pandas handles nullable/timestamp values and avoids a huge object-list copy.
        frame.to_json(self.directory / f"{name}.json", orient="records", force_ascii=False,
                      indent=2, date_format="iso", double_precision=15)
        return frame

    def object(self, name, value):
        save_json(self.directory / f"{name}.json", value)
        rows = [{"key": key, "value": json.dumps(clean(item), ensure_ascii=False, allow_nan=False)}
                for key, item in value.items()]
        pd.DataFrame(rows).to_csv(self.directory / f"{name}.csv", index=False, encoding="utf-8-sig")
