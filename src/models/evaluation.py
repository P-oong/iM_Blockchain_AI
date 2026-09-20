"""Evaluation of BAD probabilities and validation-fitted risk grades.

All probabilities mean P(BAD=1). Grade boundaries are learned once from the
validation set and reused unchanged on later samples. No function fits on OOT.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _probabilities(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Probabilities must be a finite one-dimensional array.")
    if ((array < 0) | (array > 1)).any():
        raise ValueError("Probabilities must lie between zero and one.")
    return array


def _outcomes(y: Iterable[int], p: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _probabilities(p)
    labels = np.asarray(y, dtype=float)
    if labels.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("Outcomes and probabilities must have the same 1-D shape.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Outcomes must contain only BAD=0 or BAD=1, without NA.")
    return labels, probabilities


def classification_metrics(y: Iterable[int], p: Iterable[float]) -> dict:
    """Binary metrics without threshold selection; undefined metrics are None.

    PR-AUC is trapezoidal area; average_precision uses the step integral. They
    are distinct, especially for small samples or many tied probabilities.
    KS is the unsigned maximum CDF separation; ROC-AUC establishes direction.
    A one-class sample retains Brier/log loss but no discrimination metrics.
    """
    labels, probabilities = _outcomes(y, p)
    n = len(labels)
    bad = int(labels.sum())
    result = {
        "n": n, "bad_count": bad,
        "bad_rate": float(bad / n) if n else None,
        "roc_auc": None, "ks": None, "pr_auc": None,
        "average_precision": None, "brier": None, "log_loss": None,
    }
    if not n:
        return result
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    result["brier"] = float(np.mean((labels - probabilities) ** 2))
    result["log_loss"] = float(-np.mean(
        labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)
    ))
    if bad == 0 or bad == n:
        return result
    order = np.argsort(-probabilities, kind="stable")
    sorted_p, sorted_y = probabilities[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), n - 1]
    true_positive = np.cumsum(sorted_y)[ends]
    false_positive = (ends + 1) - true_positive
    recall = np.r_[0.0, true_positive / bad]
    false_rate = np.r_[0.0, false_positive / (n - bad)]
    precision = np.r_[1.0, true_positive / (ends + 1)]
    # Explicit trapezoids also support NumPy versions without np.trapezoid.
    result["roc_auc"] = float(np.sum(np.diff(false_rate) * (recall[:-1] + recall[1:]) / 2))
    result["ks"] = float(np.max(np.abs(recall - false_rate)))
    result["pr_auc"] = float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) / 2))
    result["average_precision"] = float(np.sum(np.diff(recall) * precision[1:]))
    return result


def _quantile_groups(probabilities: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    if not len(probabilities):
        return np.array([], dtype=int)
    cuts = np.unique(np.quantile(probabilities, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(cuts, probabilities, side="right")


def calibration_table(y: Iterable[int], p: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    """Quantile calibration groups; tied probabilities are never split."""
    labels, probabilities = _outcomes(y, p)
    columns = ["bin", "p_min", "p_max", "count", "bad_count", "share",
               "mean_p_bad", "actual_bad_rate", "calibration_gap"]
    groups = _quantile_groups(probabilities, n_bins)
    rows = []
    for index, group in enumerate(np.unique(groups), 1):
        mask = groups == group
        actual = float(labels[mask].mean())
        predicted = float(probabilities[mask].mean())
        rows.append({
            "bin": index, "p_min": float(probabilities[mask].min()),
            "p_max": float(probabilities[mask].max()), "count": int(mask.sum()),
            "bad_count": int(labels[mask].sum()), "share": float(mask.mean()),
            "mean_p_bad": predicted, "actual_bad_rate": actual,
            "calibration_gap": actual - predicted,
        })
    return pd.DataFrame(rows, columns=columns)


def score_points(p: Iterable[float], base_score: float = 600,
                 base_odds: float = 20, pdo: float = 50) -> np.ndarray:
    """Good:bad odds doubling adds PDO; base_odds=20 means p_bad=1/21.

    Endpoints are clipped to [1e-15, 1-1e-15] to keep exported scores finite.
    The score scale is not bounded to 0..100.
    """
    probabilities = _probabilities(p)
    if not np.isfinite([base_score, base_odds, pdo]).all() or base_odds <= 0 or pdo <= 0:
        raise ValueError("Score scaling requires finite values and positive odds/PDO.")
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return base_score + pdo / math.log(2) * (
        np.log1p(-clipped) - np.log(clipped) - math.log(base_odds)
    )


def _merge_blocks(left: dict, right: dict) -> dict:
    return {
        "min_p": left["min_p"], "max_p": right["max_p"],
        "count": left["count"] + right["count"],
        "bad": left["bad"] + right["bad"],
        "sum_p": left["sum_p"] + right["sum_p"],
    }


def _rate(block: dict) -> float:
    return block["bad"] / block["count"]


def _merge_cost(left: dict, right: dict) -> float:
    return (left["count"] * right["count"] / (left["count"] + right["count"])) * (
        _rate(left) - _rate(right)
    ) ** 2


def fit_grades(y: Iterable[int], p: Iterable[float], max_grades: int = 5,
               min_fraction: float = .05, min_bad: int = 10) -> dict:
    """Fit adjacent grades to validation BAD rates using PAVA and merging.

    Start with at most 20 quantile bins. Merge decreasing/equal empirical bad
    rates, then under-supported adjacent bins, then least-distinct adjacent
    rates until at most max_grades remain. The result may have fewer grades.
    These are candidate grades for policy review, not externally validated
    risk cutoffs. A single grade with insufficient events is explicitly flagged.
    """
    labels, probabilities = _outcomes(y, p)
    if not len(labels):
        raise ValueError("Grade fitting requires nonempty validation data.")
    if max_grades < 1 or not 0 < min_fraction <= 1 or min_bad < 0:
        raise ValueError("Invalid grade support constraints.")
    groups = _quantile_groups(probabilities, max(20, max_grades))
    blocks = []
    for group in np.unique(groups):
        mask = groups == group
        blocks.append({
            "min_p": float(probabilities[mask].min()),
            "max_p": float(probabilities[mask].max()),
            "count": int(mask.sum()), "bad": int(labels[mask].sum()),
            "sum_p": float(probabilities[mask].sum()),
        })
    # Pool-adjacent-violators: equal rates also offer no observed risk separation.
    pooled = []
    for block in blocks:
        pooled.append(block)
        while len(pooled) > 1 and _rate(pooled[-2]) >= _rate(pooled[-1]):
            pooled[-2:] = [_merge_blocks(pooled[-2], pooled[-1])]
    blocks = pooled
    minimum_count = math.ceil(len(labels) * min_fraction)
    while len(blocks) > 1:
        small = [i for i, block in enumerate(blocks)
                 if block["count"] < minimum_count or block["bad"] < min_bad]
        if not small:
            break
        index = min(small, key=lambda i: (blocks[i]["count"], i))
        adjacent = [i for i in (index - 1, index) if 0 <= i < len(blocks) - 1]
        boundary = min(adjacent, key=lambda i: (_merge_cost(blocks[i], blocks[i + 1]), i))
        blocks[boundary:boundary + 2] = [_merge_blocks(blocks[boundary], blocks[boundary + 1])]
    while len(blocks) > max_grades:
        boundary = min(range(len(blocks) - 1),
                       key=lambda i: (_merge_cost(blocks[i], blocks[i + 1]), i))
        blocks[boundary:boundary + 2] = [_merge_blocks(blocks[boundary], blocks[boundary + 1])]
    cuts = []
    for left, right in zip(blocks[:-1], blocks[1:]):
        cut = left["max_p"] + (right["min_p"] - left["max_p"]) / 2
        # Adjacent IEEE floats may have no representable midpoint.
        cuts.append(cut if cut > left["max_p"] else right["min_p"])
    warnings = []
    if len(blocks) < max_grades:
        warnings.append(f"Validation supports {len(blocks)} grades; requested maximum was {max_grades}.")
    if len(np.unique(labels)) < 2:
        warnings.append("Validation has one outcome class; risk separation cannot be assessed.")
    if any(block["bad"] < min_bad for block in blocks):
        warnings.append("Total validation events cannot satisfy minimum_bad; single grade is provisional.")
    table = []
    for index, block in enumerate(blocks):
        table.append({
            "grade": f"R{index + 1}", "p_lower": 0.0 if index == 0 else cuts[index - 1],
            "p_upper": 1.0 if index == len(blocks) - 1 else cuts[index],
            "count": block["count"], "share": block["count"] / len(labels),
            "bad_count": block["bad"], "actual_bad_rate": _rate(block),
            "mean_p_bad": block["sum_p"] / block["count"],
            "support_ok": block["count"] >= minimum_count and block["bad"] >= min_bad,
        })
    return {
        "method": "validation_quantiles_pava_adjacent_merge",
        "probability_cuts": cuts, "grades": [row["grade"] for row in table],
        "boundary_rule": "lower_inclusive_upper_exclusive_except_last",
        "validation_table": table, "warnings": warnings,
        "constraints": {"max_grades": max_grades, "min_fraction": min_fraction,
                        "min_bad": min_bad, "min_count": minimum_count},
    }


def assign_grades(p: Iterable[float], definition: dict) -> np.ndarray:
    """Apply frozen validation cuts; equality to a cut goes to the riskier grade."""
    probabilities = _probabilities(p)
    cuts = np.asarray(definition["probability_cuts"], dtype=float)
    grades = np.asarray(definition["grades"], dtype=str)
    if (cuts.ndim != 1 or not np.isfinite(cuts).all() or
            (np.diff(cuts) <= 0).any() or ((cuts < 0) | (cuts > 1)).any() or
            len(grades) != len(cuts) + 1):
        raise ValueError("Invalid frozen grade definition.")
    return grades[np.searchsorted(cuts, probabilities, side="right")]


def population_stability(reference_labels: Iterable, current_labels: Iterable,
                         smoothing: float = .5) -> tuple[float, pd.DataFrame]:
    """PSI over fixed bin labels, adding smoothing to both counts in every bin.

    Numeric inputs must first be binned using TRAIN cuts. This function does
    not refit bins. The union includes unseen current labels and missing bins.
    """
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("PSI smoothing must be finite and positive.")
    def counts(values):
        # Prefixing observed labels keeps a literal '<MISSING>' category distinct.
        return pd.Series(list(values), dtype=object).map(
            lambda value: "missing:" if pd.isna(value) else f"value:{value}"
        ).value_counts(sort=False)
    reference = counts(reference_labels)
    current = counts(current_labels)
    if not reference.sum() or not current.sum():
        raise ValueError("PSI requires nonempty reference and current populations.")
    bins = sorted(set(reference.index) | set(current.index))
    reference_count = reference.reindex(bins, fill_value=0).to_numpy(dtype=int)
    current_count = current.reindex(bins, fill_value=0).to_numpy(dtype=int)
    reference_share = (reference_count + smoothing) / (reference_count.sum() + smoothing * len(bins))
    current_share = (current_count + smoothing) / (current_count.sum() + smoothing * len(bins))
    parts = (current_share - reference_share) * np.log(current_share / reference_share)
    table = pd.DataFrame({
        "bin": bins, "reference_count": reference_count, "current_count": current_count,
        "reference_share": reference_share, "current_share": current_share,
        "psi_component": parts,
    })
    return float(parts.sum()), table


def episode_variance(scores: Iterable[float], episodes: Iterable) -> dict:
    """Population variance = within-episode + between-episode variance."""
    frame = pd.DataFrame({"score": np.asarray(scores, dtype=float), "episode": list(episodes)})
    if frame.empty or not np.isfinite(frame["score"]).all() or frame["episode"].isna().any():
        raise ValueError("Finite nonempty scores and complete episode IDs are required.")
    means = frame.groupby("episode", sort=False)["score"].transform("mean")
    total = float(frame["score"].var(ddof=0))
    within = float(np.mean((frame["score"] - means) ** 2))
    between = float(np.mean((means - frame["score"].mean()) ** 2))
    return {"n": len(frame), "episode_count": int(frame["episode"].nunique()),
            "total_variance": total, "within_episode_variance": within,
            "between_episode_variance": between,
            "within_episode_share": within / total if total else None,
            "between_episode_share": between / total if total else None}


def contribution_variance(individual: Iterable[float], market: Iterable[float]) -> dict:
    """Decompose additive score variance, explicitly retaining covariance.

    Covariance allocation shares split 2*Cov equally between groups. They can
    be negative and should not be presented as causal importance percentages.
    """
    left, right = np.asarray(individual, dtype=float), np.asarray(market, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not len(left):
        raise ValueError("Contributions must have matching, nonempty 1-D shapes.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Contributions must be finite.")
    variance_left, variance_right = float(left.var()), float(right.var())
    covariance = float(np.mean((left - left.mean()) * (right - right.mean())))
    total = float((left + right).var())
    return {
        "total_variance": total, "individual_variance": variance_left,
        "market_variance": variance_right, "covariance": covariance,
        "twice_covariance": 2 * covariance,
        "individual_allocated_share": (variance_left + covariance) / total if total else None,
        "market_allocated_share": (variance_right + covariance) / total if total else None,
        "interpretation": "Descriptive additive-score variance allocation; not causal importance.",
    }
