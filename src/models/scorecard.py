"""Portable JSON scorecard: frozen WOE, logistic, positive sigmoid and grades."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .artifacts import save_json
from .evaluation import assign_grades
from .woe import WOEEncoder


INDIVIDUAL_FEATURES = {"업력_일수", "업력_연수", "업력_개월수", "최근1년_신규여부",
                       "최근3년_신규여부", "지역대비_업력비율_lag1q"}


class ScorecardModel:
    def __init__(self, encoder, features, coefficients, intercept, calibration,
                 grade_definition, scaling, horizon_months=12):
        self.features = list(features)
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.coefficients.shape != (len(self.features),) or not np.isfinite(self.coefficients).all():
            raise ValueError("Coefficients must match selected features")
        # Scoring requires only selected inputs, not all rejected training candidates.
        payload = encoder.to_dict()
        payload["feature_names"] = self.features
        payload["features"] = {key: payload["features"][key] for key in self.features}
        self.encoder = WOEEncoder.from_dict(payload)
        self.intercept = float(intercept)
        self.calibration = dict(calibration)
        self.grade_definition = dict(grade_definition)
        self.scaling = dict(scaling)
        self.horizon_months = horizon_months
        parameters = [self.intercept, self.calibration["slope"], self.calibration["intercept"],
                      self.scaling["base_score"], self.scaling["base_odds"], self.scaling["pdo"]]
        if not np.isfinite(parameters).all():
            raise ValueError("Model and scaling parameters must be finite")
        if self.scaling["base_odds"] <= 0 or self.scaling["pdo"] <= 0:
            raise ValueError("PDO and base_odds must be positive")
        if self.calibration["slope"] <= 0:
            raise ValueError("Calibration slope must be positive to preserve score direction")

    @property
    def factor(self):
        return self.scaling["pdo"] / np.log(2.)

    @property
    def effective_intercept(self):
        return self.calibration["intercept"] + self.calibration["slope"] * self.intercept

    @property
    def basepoints(self):
        return (self.scaling["base_score"] - self.factor * np.log(self.scaling["base_odds"])
                - self.factor * self.effective_intercept)

    def contributions(self, frame):
        transformed = self.encoder.transform(frame)[self.features]
        result = transformed.mul(-self.factor * self.calibration["slope"] * self.coefficients)
        result["basepoints"] = self.basepoints
        return result

    def predict(self, frame):
        x = self.encoder.transform(frame)[self.features].to_numpy()
        raw_logit = x @ self.coefficients + self.intercept
        calibrated_logit = self.calibration["intercept"] + self.calibration["slope"] * raw_logit
        p = expit(calibrated_logit)
        score = self.scaling["base_score"] - self.factor * (calibrated_logit + np.log(self.scaling["base_odds"]))
        return pd.DataFrame({"p_bad_raw": expit(raw_logit), "p_bad": p, "p_survival": 1 - p,
                             "score": score, "recovery_score": 100 * (1 - p),
                             "grade": assign_grades(p, self.grade_definition)}, index=frame.index)

    def to_dict(self):
        return {"format_version": 1, "target": "BAD=1 means closure; Y=1 means survival",
                "horizon_months": self.horizon_months, "features": self.features,
                "coefficients": self.coefficients.tolist(), "intercept": self.intercept,
                "calibration": self.calibration, "grade_definition": self.grade_definition,
                "scaling": self.scaling, "woe": self.encoder.to_dict()}

    def save(self, path):
        save_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["format_version"] != 1:
            raise ValueError("Unsupported scorecard format")
        return cls(WOEEncoder.from_dict(payload["woe"]), payload["features"],
                   payload["coefficients"], payload["intercept"], payload["calibration"],
                   payload["grade_definition"], payload["scaling"], payload["horizon_months"])
