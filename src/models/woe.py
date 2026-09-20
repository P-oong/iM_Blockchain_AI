"""Deterministic, train-only WOE binning using pandas and NumPy.

The target must be 1 for bad (closure), 0 for good (survival). Numeric bins
are monotone in the *training bad rate*. ``max_bins`` excludes the separate
missing bin. Missing values absent during training and unseen categories
receive neutral WOE=0. Transform never changes fitted bins or statistics.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd


def _category_key(value):
    """Keep strings, numbers and booleans distinct in a JSON-safe mapping."""
    if isinstance(value, (bool, np.bool_)):
        return "bool:" + str(bool(value)).lower()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "number:" + repr(float(value))
    return "string:" + str(value)


def _rate(group):
    return group["bad"] / group["count"] if group["count"] else 0.0


def _merge(left, right):
    return {
        "members": left["members"] + right["members"],
        "count": left["count"] + right["count"],
        "bad": left["bad"] + right["bad"],
        "good": left["good"] + right["good"],
    }


class WOEEncoder:
    """Fit fine bins and adjacent coarse bins exclusively on training rows.

    Numeric monotone candidates are built in both directions and the one
    retaining greatest training IV is selected (increasing wins exact ties).
    Categorical levels are ordered by training bad rate, then merged.
    Minimum size/class constraints are enforced where feasible; unresolved
    constraints (including the separate missing bin) are reported explicitly.
    """

    def __init__(self, fine_bins=10, max_bins=5, min_bin_fraction=.05,
                 min_bad=20, min_good=20, smoothing=.5):
        if not isinstance(fine_bins, int) or fine_bins < 1:
            raise ValueError("fine_bins must be a positive integer")
        if not isinstance(max_bins, int) or max_bins < 1:
            raise ValueError("max_bins must be a positive integer")
        if not 0 <= min_bin_fraction <= 1:
            raise ValueError("min_bin_fraction must be between 0 and 1")
        if min_bad < 0 or min_good < 0 or smoothing <= 0:
            raise ValueError("class minima must be nonnegative; smoothing must be positive")
        self.fine_bins = fine_bins
        self.max_bins = max_bins
        self.min_bin_fraction = float(min_bin_fraction)
        self.min_bad = int(min_bad)
        self.min_good = int(min_good)
        self.smoothing = float(smoothing)

    def _statistics(self, groups):
        active = [group for group in groups if group["count"] > 0]
        total_bad = sum(group["bad"] for group in active)
        total_good = sum(group["good"] for group in active)
        size = len(active)
        result = []
        for group in groups:
            record = dict(group)
            if group["count"]:
                bad_dist = (group["bad"] + self.smoothing) / (total_bad + self.smoothing * size)
                good_dist = (group["good"] + self.smoothing) / (total_good + self.smoothing * size)
                woe = math.log(bad_dist / good_dist)
                iv = (bad_dist - good_dist) * woe
            else:
                bad_dist = good_dist = woe = iv = 0.0
            record.update(bad_rate=_rate(group) if group["count"] else None,
                          bad_dist=bad_dist, good_dist=good_dist, woe=woe,
                          iv_component=iv)
            result.append(record)
        return result

    def _iv(self, groups, missing):
        return sum(row["iv_component"] for row in self._statistics(groups + [missing]))

    def _fails(self, group, minimum):
        return (group["count"] < minimum or group["bad"] < self.min_bad
                or group["good"] < self.min_good)

    def _coarsen(self, groups, missing, minimum, direction):
        groups = copy.deepcopy(groups)
        # Pool adjacent violators; count-weighted rates preserve exact totals.
        stack = []
        for group in groups:
            stack.append(group)
            while len(stack) > 1:
                difference = _rate(stack[-1]) - _rate(stack[-2])
                if (direction == "increasing" and difference < -1e-12) or (
                        direction == "decreasing" and difference > 1e-12):
                    stack[-2:] = [_merge(stack[-2], stack[-1])]
                else:
                    break
        groups = stack
        while len(groups) > 1:
            sparse = [i for i, group in enumerate(groups) if self._fails(group, minimum)]
            if sparse:
                position = min(sparse, key=lambda i: (groups[i]["count"], i))
                candidates = [i for i in (position - 1, position) if 0 <= i < len(groups) - 1]
            elif len(groups) > self.max_bins:
                candidates = list(range(len(groups) - 1))
            else:
                break
            # Keep the adjacent merge retaining the most IV, with stable ties.
            choices = []
            for position in candidates:
                merged = groups[:position] + [_merge(groups[position], groups[position + 1])] + groups[position + 2:]
                choices.append((self._iv(merged, missing), -position, merged))
            groups = max(choices, key=lambda item: (item[0], item[1]))[2]
        return groups

    def fit(self, X, y, categorical=None):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X must be a nonempty DataFrame")
        if not X.columns.is_unique or not all(isinstance(name, str) for name in X.columns):
            raise ValueError("X must have unique string column names")
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X) or not np.isin(target, [0, 1]).all():
            raise ValueError("y must contain one nonmissing binary target per X row")
        if len(np.unique(target)) != 2:
            raise ValueError("Training requires both good (0) and bad (1) outcomes")
        target = target.astype(int)
        if categorical is None:
            categorical = [name for name in X if not pd.api.types.is_numeric_dtype(X[name])]
        categorical = set(categorical)
        if categorical - set(X.columns):
            raise ValueError("categorical contains columns not present in X")
        self.feature_names_ = list(X.columns)
        self.features_ = {}
        minimum = max(1, math.ceil(self.min_bin_fraction * len(X)))
        for name in self.feature_names_:
            series = X[name]
            kind = "categorical" if name in categorical else "numeric"
            rare_count = 0
            fine_definitions = []
            if kind == "numeric":
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                missing_mask = ~np.isfinite(values)
                observed = values[~missing_mask]
                if len(observed):
                    distinct = np.unique(observed)
                    if len(distinct) <= self.fine_bins:
                        cuts = distinct[:-1].tolist()
                    else:
                        quantiles = np.quantile(observed, np.linspace(0, 1, self.fine_bins + 1)[1:-1])
                        # Snap cuts down to observed values. Every interval then
                        # contains training rows, including heavily tied data.
                        positions = np.searchsorted(distinct, quantiles, side="right") - 1
                        cuts = np.unique(distinct[np.clip(positions, 0, len(distinct) - 2)]).tolist()
                    positions = np.searchsorted(cuts, observed, side="left")
                    for index in range(len(cuts) + 1):
                        selected = positions == index
                        count = int(selected.sum())
                        if not count:
                            continue
                        bad = int(target[~missing_mask][selected].sum())
                        fine_definitions.append({"bin_id": f"F{index:02d}",
                            "lower": cuts[index - 1] if index else None,
                            "upper": cuts[index] if index < len(cuts) else None,
                            "categories": [], "count": count, "bad": bad, "good": count - bad})
                else:
                    cuts = []
            else:
                missing_mask = series.isna().to_numpy()
                keys = series[~missing_mask].map(_category_key).to_numpy()
                counts = pd.DataFrame({"key": keys, "bad": target[~missing_mask]}).groupby("key", sort=True)["bad"].agg(["count", "sum"])
                categories = []
                rare = []
                for key, row in counts.iterrows():
                    item = {"categories": [str(key)], "count": int(row["count"]),
                            "bad": int(row["sum"]), "good": int(row["count"] - row["sum"])}
                    (rare if item["count"] < minimum else categories).append(item)
                rare_count = sum(len(item["categories"]) for item in rare)
                if rare:
                    categories.append({"categories": [key for item in rare for key in item["categories"]],
                                       "count": sum(item["count"] for item in rare),
                                       "bad": sum(item["bad"] for item in rare),
                                       "good": sum(item["good"] for item in rare)})
                categories.sort(key=lambda item: (_rate(item), item["categories"]))
                for index, item in enumerate(categories):
                    fine_definitions.append(dict(item, bin_id=f"F{index:02d}", lower=None, upper=None))
                cuts = []
            missing_count = int(missing_mask.sum())
            missing_bad = int(target[missing_mask].sum())
            missing = {"members": ["MISSING"], "count": missing_count,
                       "bad": missing_bad, "good": missing_count - missing_bad}
            groups = [{"members": [item["bin_id"]], "count": item["count"],
                       "bad": item["bad"], "good": item["good"]} for item in fine_definitions]
            ascending = self._coarsen(groups, missing, minimum, "increasing")
            direction = "increasing"
            coarse = ascending
            if kind == "numeric":
                descending = self._coarsen(groups, missing, minimum, "decreasing")
                if self._iv(descending, missing) > self._iv(ascending, missing) + 1e-12:
                    coarse, direction = descending, "decreasing"
            fine_lookup = {item["bin_id"]: item for item in fine_definitions}
            records = []
            for index, group in enumerate(coarse):
                members = [fine_lookup[key] for key in group["members"]]
                records.append(dict(group, bin_id=f"B{index:02d}",
                                    lower=members[0]["lower"], upper=members[-1]["upper"],
                                    categories=[key for item in members for key in item["categories"]]))
            records.append(dict(missing, bin_id="MISSING", lower=None, upper=None, categories=[]))
            records.append({"bin_id": "UNSEEN", "members": [], "count": 0, "bad": 0,
                            "good": 0, "lower": None, "upper": None, "categories": []})
            fine_records = [dict(item, members=[item["bin_id"]]) for item in fine_definitions]
            fine_records.append(dict(missing, bin_id="MISSING", lower=None, upper=None, categories=[]))
            coarse_records = self._statistics(records)
            violation_bins = [row["bin_id"] for row in coarse_records if row["count"] and self._fails(row, minimum)]
            flags = []
            if len(coarse) < 2:
                flags.append("no_regular_bin_separation")
            if violation_bins:
                flags.append("minimum_constraints_unmet:" + ",".join(violation_bins))
            if missing_count == 0:
                flags.append("unobserved_missing_uses_neutral_woe")
            if kind == "categorical":
                flags.append("unseen_category_uses_neutral_woe")
            if kind == "numeric" and len(observed) != series.notna().sum():
                flags.append("nonfinite_or_unparseable_numeric_treated_as_missing")
            self.features_[name] = {
                "kind": kind, "direction": direction, "fine_cuts": cuts,
                "fine": self._statistics(fine_records), "coarse": coarse_records,
                "summary": {"feature": name, "kind": kind,
                    "iv": float(sum(row["iv_component"] for row in coarse_records)),
                    "n_bins": len(coarse) + int(missing_count > 0), "n_regular_bins": len(coarse),
                    "monotonic": True, "direction": direction,
                    "missing_count": missing_count, "missing_fraction": missing_count / len(X),
                    "rare_category_count": rare_count, "min_required_count": minimum,
                    "min_bin_count": min((row["count"] for row in coarse_records if row["count"]), default=0),
                    "constraint_violations": len(violation_bins), "constant": len(coarse) + int(missing_count > 0) < 2,
                    "fit_rows": len(X), "audit_flags": ";".join(flags)},
            }
        return self

    def _check_fitted(self):
        if not hasattr(self, "features_"):
            raise ValueError("WOEEncoder has not been fitted")

    def bin_ids(self, X):
        self._check_fitted()
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        missing_columns = set(self.feature_names_) - set(X.columns)
        if missing_columns:
            raise ValueError("Missing fitted columns: " + ", ".join(sorted(missing_columns)))
        output = {}
        for name in self.feature_names_:
            specification = self.features_[name]
            series = X[name]
            identifiers = np.full(len(series), "MISSING", dtype=object)
            regular = [row for row in specification["coarse"] if row["bin_id"] not in {"MISSING", "UNSEEN"}]
            if specification["kind"] == "numeric":
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                valid = np.isfinite(values)
                if regular:
                    cuts = [row["upper"] for row in regular[:-1]]
                    positions = np.searchsorted(cuts, values[valid], side="left")
                    identifiers[valid] = np.array([row["bin_id"] for row in regular], dtype=object)[positions]
                else:
                    identifiers[valid] = "UNSEEN"
            else:
                mapping = {key: row["bin_id"] for row in regular for key in row["categories"]}
                valid = ~series.isna().to_numpy()
                identifiers[valid] = [mapping.get(_category_key(value), "UNSEEN") for value in series[valid]]
            output[name] = identifiers
        return pd.DataFrame(output, index=X.index)

    def transform(self, X):
        identifiers = self.bin_ids(X)
        return pd.DataFrame({name: identifiers[name].map({row["bin_id"]: row["woe"]
                            for row in self.features_[name]["coarse"]}).astype(float)
                             for name in self.feature_names_}, index=X.index)

    def tables(self):
        self._check_fitted()
        tables = []
        for stage in ("fine", "coarse"):
            rows = []
            for feature in self.feature_names_:
                specification = self.features_[feature]
                for record in specification[stage]:
                    row = copy.deepcopy(record)
                    if row["bin_id"] in {"MISSING", "UNSEEN"}:
                        definition = row["bin_id"]
                    elif specification["kind"] == "numeric":
                        lower = "-inf" if row["lower"] is None else str(row["lower"])
                        upper = "+inf" if row["upper"] is None else str(row["upper"])
                        definition = f"({lower}, {upper}]"
                    else:
                        definition = json.dumps(row["categories"], ensure_ascii=False)
                    row.update(feature=feature, kind=specification["kind"], stage=stage,
                               direction=specification["direction"], definition=definition,
                               count_fraction=row["count"] / specification["summary"]["fit_rows"],
                               audit_flags=specification["summary"]["audit_flags"])
                    row["categories"] = json.dumps(row["categories"], ensure_ascii=False)
                    row["members"] = json.dumps(row["members"], ensure_ascii=False)
                    rows.append(row)
            tables.append(pd.DataFrame(rows))
        return tuple(tables)

    def summary(self):
        self._check_fitted()
        return pd.DataFrame([copy.deepcopy(self.features_[name]["summary"]) for name in self.feature_names_])

    def to_dict(self):
        self._check_fitted()
        return {"schema_version": 1, "target": "1=bad,0=good", "woe_convention": "ln(bad_dist/good_dist)",
                "parameters": {name: getattr(self, name) for name in ("fine_bins", "max_bins", "min_bin_fraction", "min_bad", "min_good", "smoothing")},
                "feature_names": list(self.feature_names_), "features": copy.deepcopy(self.features_)}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported WOE schema version")
        encoder = cls(**payload["parameters"])
        encoder.feature_names_ = list(payload["feature_names"])
        encoder.features_ = copy.deepcopy(payload["features"])
        return encoder
