"""Reproducible, time-purged WOE logistic scorecard development and reporting.

Run from repository root: python -m src.models.train_scorecard
OOТ labels are accessed for evaluation only after coefficients/calibration/grades freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .artifacts import Reports, save_json
from .config import ScorecardConfig
from .evaluation import (assign_grades, calibration_table, classification_metrics,
                         contribution_variance, episode_variance, fit_grades,
                         population_stability, score_points)
from .scorecard import INDIVIDUAL_FEATURES, ScorecardModel
from .plots import save_figures
from .woe import WOEEncoder


def announce(message):
    print(message, flush=True)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def forbidden_feature(name):
    return (name in {"관리번호", "t0", "기준분기", "crisis_flag", "episode_start", "읍면동",
                     "행정동코드", "위도", "경도", "지하철_최근접역", "서비스인구_매핑동"}
            or name.startswith(("Y_", "BAD", "horizon_end", "label_status"))
            or any(word in name.lower() for word in ("synthetic", "scenario", "충격계수", "충격유형")))


def load_source(config, reports):
    """Exact-row deduplication before sampling; conflicts never silently averaged."""
    metadata = json.loads(Path(config.metadata).read_text(encoding="utf-8-sig"))
    header = pd.read_csv(config.source, nrows=0, encoding="utf-8-sig").columns.tolist()
    target = f"Y_{config.horizon_months}M"
    end = f"horizon_end_{config.horizon_months}M"
    status = f"label_status_{config.horizon_months}M"
    required = {"관리번호", "t0", "기준분기", "구", "읍면동", "업종", "crisis_flag", "episode_start", target, end, status}
    if required - set(header):
        raise ValueError(f"Missing required columns: {sorted(required - set(header))}. Labels are not inferred from another horizon.")
    base = metadata["feature_allowlist"]
    absent = set(base) - set(header)
    if absent:
        raise ValueError(f"Allowlist columns missing: {sorted(absent)}")
    candidates, inventory = [], []
    for name in header:
        if forbidden_feature(name):
            reason = "excluded_identifier_target_geography_or_generation_metadata"
        elif name == "구" and not config.include_district:
            reason = "excluded_district_experiment"
        elif name in config.excluded_features:
            reason = "excluded_by_config"
        elif name in base:
            reason = "candidate_labeling_allowlist"
            candidates.append(name)
        elif name in config.extra_feature_availability:
            reason = "candidate_extra_with_availability_check"
            candidates.append(name)
        else:
            reason = "excluded_unverified_enrichment_or_metadata"
        inventory.append({"feature": name, "decision": reason})
    reports.table("feature_inventory", inventory)
    # Chunking keeps the 550 MB enriched input from needing several GB at once.
    seen = set()
    parts, raw_rows, raw_bad, raw_labeled = [], 0, 0, 0
    keep = sorted(required | set(candidates) | {"행정동코드"} |
                  set(config.extra_feature_availability.values()))
    keep = [name for name in keep if name in header]
    for chunk in pd.read_csv(config.source, encoding="utf-8-sig", chunksize=15000, dtype="string"):
        raw_rows += len(chunk)
        outcome = pd.to_numeric(chunk[target], errors="coerce")
        raw_bad += int(outcome.eq(0).sum())
        raw_labeled += int(outcome.notna().sum())
        # Two independent pandas hashes guard practical hash collision risk.
        a = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        b = pd.util.hash_pandas_object(chunk.iloc[:, ::-1], index=False).to_numpy()
        mask = []
        for pair in zip(a, b):
            key = (int(pair[0]), int(pair[1]))
            fresh = key not in seen
            mask.append(fresh)
            seen.add(key)
        parts.append(chunk.loc[mask, keep])
    frame = pd.concat(parts, ignore_index=True)
    if frame["관리번호"].isna().any():
        raise ValueError("Missing management IDs")
    if frame["관리번호"].duplicated().any():
        raise ValueError("Conflicting or non-identical rows per management ID. Resolve joins before fitting.")
    frame["t0"] = pd.to_datetime(frame["t0"], errors="raise")
    frame["horizon_end"] = pd.to_datetime(frame[end], errors="raise")
    if frame[["t0", "horizon_end"]].isna().any().any():
        raise ValueError("Missing baseline/horizon date")
    if not frame["t0"].dt.is_quarter_end.all():
        raise ValueError("t0 must be quarter end")
    expected_end = frame["t0"] + pd.offsets.MonthEnd(config.horizon_months)
    if not frame["horizon_end"].eq(expected_end).all():
        raise ValueError("Outcome-window endpoints differ from configured month-end horizon")
    if not frame["기준분기"].eq(frame["t0"].dt.to_period("Q").astype(str)).all():
        raise ValueError("Quarter/t0 mismatch")
    for flag in ("crisis_flag", "episode_start"):
        frame[flag] = pd.to_numeric(frame[flag], errors="raise")
    if not (frame["crisis_flag"].eq(1) & frame["episode_start"].eq(1)).all():
        raise ValueError("Only crisis episode-start cohort rows are supported")
    frame[target] = pd.to_numeric(frame[target], errors="raise")
    if not frame[target].dropna().isin([0, 1]).all():
        raise ValueError("Survival target must contain only 0/1/NA")
    mature = frame["horizon_end"].le(pd.Timestamp(config.observation_end))
    frame["eligible"] = mature & frame[target].notna() & frame[status].eq("observed")
    frame["BAD"] = 1 - frame[target]
    for feature, date_column in config.extra_feature_availability.items():
        if forbidden_feature(feature) or feature not in header or date_column not in frame:
            raise ValueError(f"Extra feature/availability definition invalid: {feature}")
        available = pd.to_datetime(frame[date_column], errors="coerce")
        if available.isna().any() or (available > frame["t0"]).any():
            raise ValueError(f"{feature}: availability must be known and <= t0 for every row")
    for feature in candidates:
        if feature not in ("구", "업종"):
            frame[feature] = pd.to_numeric(frame[feature], errors="raise").replace([np.inf, -np.inf], np.nan)
    frame = frame.sort_values(["t0", "관리번호"], kind="stable").reset_index(drop=True)
    frame["episode_id"] = frame[["구", "읍면동", "업종", "기준분기"]].astype(str).agg("|".join, axis=1)
    audit = {"source_rows": raw_rows, "exact_duplicate_rows_removed": raw_rows - len(frame),
             "unique_businesses": len(frame), "raw_weighted_bad_rate": raw_bad / raw_labeled if raw_labeled else None,
             "deduplicated_bad_rate": float(frame.loc[frame.eligible, "BAD"].mean()),
             "eligible_rows": int(frame.eligible.sum()), "ineligible_rows": int((~frame.eligible).sum()),
             "t0_min": str(frame.t0.min().date()), "t0_max": str(frame.t0.max().date()),
             "episode_count": int(frame.episode_id.nunique()), "candidate_count": len(candidates),
             "input_columns": len(header), "unavailable_targets": [f"Y_{h}M" for h in (12,18,24) if f"Y_{h}M" not in header],
             "observation_end_assumption": config.observation_end,
             "deduplication": "two 64-bit full-row hashes; non-identical ID conflicts fail closed"}
    reports.object("data_audit", audit)
    return frame, candidates, audit


def assign_splits(frame, config):
    t, end = frame["t0"], frame["horizon_end"]
    result = pd.Series("ineligible", index=frame.index, dtype=object)
    eligible = frame["eligible"]
    result.loc[eligible] = "embargo"
    result.loc[eligible & (t < config.valid_start) & (end < config.valid_start)] = "train"
    result.loc[eligible & (t >= config.valid_start) & (t < config.valid_end) & (end < config.oot_start)] = "validation"
    result.loc[eligible & (t >= config.oot_start) & (t < config.oot_end)] = "oot"
    return result


def fit_logistic(x, y, c, config):
    model = LogisticRegression(penalty="l1", C=c, solver="liblinear", class_weight=None,
                               random_state=config.random_seed, max_iter=config.max_iter, tol=1e-7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return model, converged


def fit_sigmoid(logits, y):
    """Positive Platt slope preserves ranking and an additive scorecard."""
    logits, y = np.asarray(logits, float), np.asarray(y, float)
    def objective(parameters):
        intercept, slope = parameters
        z = intercept + slope * logits
        residual = expit(z) - y
        return (float(np.mean(np.logaddexp(0, z) - y * z)),
                np.array([residual.mean(), np.mean(residual * logits)]))
    fitted = minimize(objective, [0., 1.], jac=True, method="L-BFGS-B", bounds=[(-20,20),(.01,10)])
    if not fitted.success:
        raise ValueError(f"Calibration optimizer failed: {fitted.message}")
    return {"method": "sigmoid", "intercept": float(fitted.x[0]), "slope": float(fitted.x[1])}


def choose_calibration(logits, valid, reports):
    identity = {"method": "none", "intercept": 0., "slope": 1.}
    quarters = sorted(valid["t0"].unique())
    rows = []
    if len(quarters) < 2:
        reports.table("calibration_comparison", [{"method": "none", "selected": True,
                      "reason": "not_enough_validation_quarters_for_independent_calibration_check"}])
        return identity
    cut = quarters[len(quarters) // 2]
    early = (valid.t0 < cut).to_numpy()
    late = ~early
    y = valid.BAD.to_numpy(dtype=int)
    if len(np.unique(y[early])) < 2 or len(np.unique(y[late])) < 2:
        reports.table("calibration_comparison", [{"method": "none", "selected": True,
                      "reason": "one_class_validation_calibration_part"}])
        return identity
    sigmoid = fit_sigmoid(logits[early], y[early])
    raw = classification_metrics(y[late], expit(logits[late]))
    adjusted = classification_metrics(y[late], expit(sigmoid["intercept"] + sigmoid["slope"] * logits[late]))
    improves = adjusted["brier"] < raw["brier"] and adjusted["log_loss"] < raw["log_loss"]
    for method, statistics in [("none", raw), ("sigmoid", adjusted)]:
        rows.append({"method": method, "evaluation": "late_validation_quarters",
                     "fit_end": str(valid.loc[early, "t0"].max().date()),
                     "check_start": str(valid.loc[late, "t0"].min().date()),
                     "selected": method == ("sigmoid" if improves else "none"), **statistics})
    reports.table("calibration_comparison", rows)
    return fit_sigmoid(logits, y) if improves else identity


def grade_performance(frame, grades):
    rows = []
    for split, part in frame.groupby("split", sort=False):
        if split not in ("train", "validation", "oot"):
            continue
        for grade in grades:
            group = part[part.grade.eq(grade)]
            rows.append({"split": split, "grade": grade, "count": len(group),
                         "share": len(group)/len(part), "bad_count": int(group.BAD.sum()),
                         "actual_bad_rate": float(group.BAD.mean()) if len(group) else None,
                         "mean_p_bad": float(group.p_bad.mean()) if len(group) else None,
                         "score_min": float(group.score.min()) if len(group) else None,
                         "score_max": float(group.score.max()) if len(group) else None})
    return pd.DataFrame(rows)


def bootstrap_metrics(oot, config):
    rng = np.random.default_rng(config.random_seed)
    clusters = list(oot.groupby("episode_id", sort=False).indices.values())
    samples = []
    for _ in range(config.cluster_bootstrap_repeats):
        index = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
        samples.append(classification_metrics(oot.BAD.to_numpy()[index], oot.p_bad.to_numpy()[index]))
    rows = []
    for metric in ("roc_auc", "ks", "pr_auc", "average_precision", "brier", "log_loss"):
        values = [x[metric] for x in samples if x[metric] is not None]
        rows.append({"metric": metric, "cluster": "market_crisis_episode", "clusters": len(clusters),
                     "replicates": len(values), "lower_95": np.quantile(values,.025) if values else None,
                     "upper_95": np.quantile(values,.975) if values else None})
    return rows


def run_pipeline(config=None):
    config = config or ScorecardConfig()
    config.validate()
    started = time.perf_counter()
    source_hash = fingerprint(config.source)
    run_id = f"{config.horizon_months}m_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{source_hash['sha256'][:8]}"
    reports = Reports(Path(config.report_root) / run_id)
    model_dir = Path(config.model_root) / run_id
    model_dir.mkdir(parents=True, exist_ok=False)
    reports.object("config", config.to_dict())
    announce("[1/8] Validate source and remove exact duplicate records")
    frame, candidates, audit = load_source(config, reports)
    frame["split"] = assign_splits(frame, config)
    reports.table("split_summary", frame.groupby("split", sort=False).agg(
        count=("관리번호","size"), bad_count=("BAD","sum"), bad_rate=("BAD","mean"),
        t0_min=("t0","min"), t0_max=("t0","max"), horizon_max=("horizon_end","max"),
        episodes=("episode_id","nunique")).reset_index())
    groups = {name: frame[frame.split.eq(name)] for name in ("train", "validation", "oot")}
    for name, group in groups.items():
        if not len(group):
            raise ValueError(f"Empty {name} after maturity/embargo. Review split dates.")
    for name in ("train", "validation"):
        if groups[name].BAD.nunique() != 2:
            raise ValueError(f"Both classes required in {name}")
    if frame.groupby("episode_id").split.nunique().max() != 1:
        raise ValueError("A crisis episode crosses split boundaries")
    train, valid = groups["train"], groups["validation"]
    announce(f"[2/8] Train-only screening and WOE ({len(train)} train, {len(valid)} validation)")
    screening, usable = [], []
    for feature in candidates:
        series = train[feature]
        missing, unique = float(series.isna().mean()), int(series.nunique())
        dominance = float(series.value_counts(dropna=False, normalize=True).max())
        reason = ("constant" if unique <= 1 else "high_missing" if missing >= config.missing_limit
                  else "dominant_value" if dominance >= config.dominant_limit else "pass")
        screening.append({"feature": feature, "missing_rate": missing, "unique_count": unique,
                          "dominant_share": dominance, "decision": reason})
        if reason == "pass":
            usable.append(feature)
    reports.table("feature_screening", screening)
    if not usable:
        raise ValueError("No variables passed data-quality screening")
    encoder = WOEEncoder(config.fine_bins, config.max_bins, config.min_bin_fraction,
                         config.min_bin_bad, config.min_bin_good, config.smoothing)
    encoder.fit(train[usable], train.BAD.astype(int), categorical=[f for f in ("업종","구") if f in usable])
    fine, coarse = encoder.tables()
    reports.table("woe_fine", fine)
    reports.table("woe_coarse", coarse)
    stats = encoder.summary().set_index("feature")
    train_woe, valid_woe = encoder.transform(train), encoder.transform(valid)
    train_bins = encoder.bin_ids(train)
    # Time stability for selection uses training years only, never validation or OOT labels.
    temporal_bins, time_psi = [], []
    for feature in usable:
        annual_psi = []
        for year, subset in train.groupby(train.t0.dt.year):
            psi, _ = population_stability(train_bins[feature], train_bins.loc[subset.index, feature])
            annual_psi.append(psi)
            time_psi.append({"feature":feature, "year":int(year), "psi":psi})
            detail = pd.DataFrame({"bin_id":train_bins.loc[subset.index,feature], "BAD":subset.BAD})
            detail = detail.groupby("bin_id").BAD.agg(["count","sum","mean"]).reset_index()
            detail.columns = ["bin_id","count","bad_count","bad_rate"]
            detail["feature"], detail["year"] = feature, int(year)
            temporal_bins.append(detail)
        stats.loc[feature,"max_train_year_psi"] = max(annual_psi)
    reports.table("training_year_psi", time_psi)
    reports.table("training_year_bins", pd.concat(temporal_bins,ignore_index=True))
    stats["high_iv_review"] = stats.iv >= config.iv_review
    reports.table("woe_summary", stats.reset_index())
    selection = [{"feature":row["feature"], "stage":"quality", "decision":row["decision"]} for row in screening]
    iv_keep = [feature for feature in usable if stats.loc[feature,"iv"] >= config.iv_min and train_woe[feature].nunique() > 1]
    for feature in usable:
        selection.append({"feature":feature, "stage":"iv", "decision":"keep" if feature in iv_keep else "drop_low_iv_or_constant_woe", "value":stats.loc[feature,"iv"]})
    if not iv_keep:
        raise ValueError("No variables satisfy IV threshold; review data instead of silently weakening threshold")
    correlation = train_woe[iv_keep].corr(method="spearman")
    reports.table("correlation_matrix", correlation.rename_axis("feature").reset_index())
    # Deterministic priority: IV, within-train time stability, documented age representation, missingness.
    priority = sorted(iv_keep, key=lambda f:(-stats.loc[f,"iv"], stats.loc[f,"max_train_year_psi"],
                                           0 if f == "업력_연수" else 1, stats.loc[f,"missing_fraction"], f))
    keep, pairs = [], []
    for feature in priority:
        conflicts = [other for other in keep if abs(correlation.loc[feature,other]) > config.correlation_limit]
        if conflicts:
            other = max(conflicts,key=lambda f:abs(correlation.loc[feature,f]))
            selection.append({"feature":feature,"stage":"correlation","decision":"drop","retained_feature":other,
                              "value":float(correlation.loc[feature,other])})
        else:
            keep.append(feature)
            selection.append({"feature":feature,"stage":"correlation","decision":"keep"})
    for i, left in enumerate(iv_keep):
        for right in iv_keep[i+1:]:
            pairs.append({"feature_a":left,"feature_b":right,"spearman":correlation.loc[left,right],
                          "above_threshold":abs(correlation.loc[left,right])>config.correlation_limit})
    reports.table("correlation_pairs", pairs)
    announce(f"[3/8] Tune L1 logistic on validation ({len(keep)} variables before L1)")
    tuning, fitted = [], {}
    x, xv = train_woe[keep].to_numpy(), valid_woe[keep].to_numpy()
    y, yv = train.BAD.to_numpy(dtype=int), valid.BAD.to_numpy(dtype=int)
    for c in config.c_grid:
        model, converged = fit_logistic(x,y,c,config)
        statistics = classification_metrics(yv, model.predict_proba(xv)[:,1])
        active = np.abs(model.coef_[0]) > 1e-8
        tuning.append({"C":c,"converged":converged,"n_features":int(active.sum()),
                       "negative_coefficients":int((model.coef_[0]<-1e-8).sum()), **statistics})
        fitted[c] = model
    search = pd.DataFrame(tuning)
    allowed = search[search.converged & search.n_features.gt(0)]
    if allowed.empty:
        raise ValueError("No converged nonconstant logistic model")
    competitive = allowed[allowed.roc_auc >= allowed.roc_auc.max()-config.auc_tolerance]
    chosen = competitive.sort_values(["brier","n_features","negative_coefficients","C"],kind="stable").iloc[0]
    model = fitted[chosen.C]
    search["selected"] = search.C.eq(chosen.C)
    reports.table("c_tuning",search)
    active = np.abs(model.coef_[0])>1e-8
    selected = [f for f,a in zip(keep,active) if a]
    coefficients = model.coef_[0][active]
    for feature, coefficient in zip(keep, model.coef_[0]):
        selection.append({"feature":feature,"stage":"l1","decision":"keep" if abs(coefficient)>1e-8 else "drop_zero_coefficient","value":coefficient})
    reports.table("selection_log",selection)
    announce("[4/8] Validation calibration and empirical grade boundaries")
    # Remove numerical near-zero coefficients consistently before calibrating/serializing.
    raw_logits = valid_woe[selected].to_numpy() @ coefficients + model.intercept_[0]
    calibration = choose_calibration(raw_logits,valid,reports)
    valid_p = expit(calibration["intercept"]+calibration["slope"]*raw_logits)
    grade_definition = fit_grades(yv, valid_p, config.max_grades, config.min_grade_fraction, config.min_grade_bad)
    scaling = {"base_score":config.base_score,"base_odds":config.base_odds,"pdo":config.pdo}
    scorecard = ScorecardModel(encoder,selected,coefficients,model.intercept_[0],calibration,grade_definition,scaling,config.horizon_months)
    reports.object("grade_definition",grade_definition)
    cut_table = pd.DataFrame(grade_definition["validation_table"])
    cut_table["score_lower_exclusive"] = score_points(cut_table.p_upper, **scaling)
    cut_table["score_upper_inclusive"] = score_points(cut_table.p_lower, **scaling)
    cut_table.loc[cut_table.p_upper.eq(1),"score_lower_exclusive"] = np.nan
    cut_table.loc[cut_table.p_lower.eq(0),"score_upper_inclusive"] = np.nan
    reports.table("grade_cutoffs",cut_table)
    coefficient_table = pd.DataFrame({"feature":selected,"coefficient":coefficients,
                                    "calibrated_coefficient":coefficients*calibration["slope"],
                                    "odds_ratio_per_woe_unit":np.exp(coefficients),
                                    "negative_sign_review":coefficients<0})
    reports.table("coefficients",coefficient_table)
    reports.object("model_parameters",{"raw_intercept":float(model.intercept_[0]),"calibration":calibration,
                   "effective_intercept":scorecard.effective_intercept,"basepoints":scorecard.basepoints,
                   "scaling":scaling,"selected_C":float(chosen.C),"selected_features":selected})
    card = coarse[coarse.feature.isin(selected)].copy()
    coefficient_map = dict(zip(selected,coefficients))
    card["coefficient"] = card.feature.map(coefficient_map)
    card["points"] = -scorecard.factor*calibration["slope"]*card.coefficient*card.woe
    card = pd.concat([pd.DataFrame([{"feature":"basepoints","bin_id":"BASE","points":scorecard.basepoints}]),card],ignore_index=True)
    reports.table("scorecard",card)
    scorecard.save(model_dir/"scorecard_model.json")
    save_json(model_dir/"config.json", config.to_dict())
    save_json(model_dir/"woe_all_candidates.json",encoder.to_dict())
    # All development choices are frozen above; OOT is evaluated only below.
    announce("[5/8] Frozen-model OOT evaluation and scoring all unique businesses")
    predictions = scorecard.predict(frame)
    scored = pd.concat([frame[["관리번호","t0","horizon_end","구","읍면동","업종","episode_id","split",f"Y_{config.horizon_months}M","BAD"]],predictions],axis=1)
    contributions = scorecard.contributions(frame)
    individual = [f for f in selected if f in INDIVIDUAL_FEATURES]
    market = [f for f in selected if f not in INDIVIDUAL_FEATURES]
    scored["individual_points"] = contributions[individual].sum(axis=1)
    scored["market_points"] = contributions[market].sum(axis=1)
    scored["basepoints"] = scorecard.basepoints
    if not np.allclose(contributions.sum(axis=1),scored.score,atol=1e-8,rtol=0):
        raise AssertionError("Score contributions do not reconstruct total score")
    replay = ScorecardModel.load(model_dir/"scorecard_model.json").predict(frame)
    if not np.allclose(replay.score,scored.score,atol=1e-10,rtol=0) or not np.array_equal(replay.grade,scored.grade):
        raise AssertionError("Portable model replay mismatch")
    reports.table("scored_businesses",scored)
    reports.table("score_contributions",pd.concat([frame[["관리번호","split"]],contributions],axis=1))
    metrics, curves, yearly = [], [], []
    baseline = float(train.BAD.mean())
    for split in ("train","validation","oot"):
        part = scored[scored.split.eq(split)]
        for kind, probabilities in (("raw",part.p_bad_raw),("final",part.p_bad),("train_prevalence_baseline",np.full(len(part),baseline))):
            metrics.append({"split":split,"model":kind,**classification_metrics(part.BAD,probabilities)})
        curve = calibration_table(part.BAD,part.p_bad)
        curve["split"] = split
        curves.append(curve)
        for year, period in part.groupby(part.t0.dt.year):
            yearly.append({"split":split,"year":int(year),**classification_metrics(period.BAD,period.p_bad)})
    metric_frame = reports.table("metrics",metrics)
    reports.table("calibration_curve",pd.concat(curves,ignore_index=True))
    reports.table("metrics_by_year",yearly)
    grade_table = reports.table("grade_performance",grade_performance(scored,grade_definition["grades"]))
    announce("[6/8] PSI, episode bootstrap and score variation attribution")
    all_bins = encoder.bin_ids(frame)
    score_cuts = np.unique(np.quantile(scored.loc[train.index,"score"], np.arange(.1,1,.1)))
    all_bins["TOTAL_SCORE"] = np.searchsorted(score_cuts,scored.score,side="right").astype(str)
    save_json(model_dir/"score_psi_cuts.json",score_cuts.tolist())
    summaries,details = [],[]
    comparisons = {name:groups[name].index for name in ("validation","oot")}
    comparisons.update({f"year_{int(year)}":part.index for year,part in frame.groupby(frame.t0.dt.year)})
    for comparison,index in comparisons.items():
        for feature in list(usable)+["TOTAL_SCORE"]:
            psi, detail = population_stability(all_bins.loc[train.index,feature],all_bins.loc[index,feature])
            summaries.append({"comparison":comparison,"feature":feature,"psi":psi,
                              "review_flag":psi>=.25,"in_final_model":feature in selected or feature=="TOTAL_SCORE"})
            detail["comparison"],detail["feature"] = comparison,feature
            details.append(detail)
    psi_frame = reports.table("psi_summary",summaries)
    reports.table("psi_detail",pd.concat(details,ignore_index=True))
    reports.table("oot_cluster_confidence_intervals",bootstrap_metrics(scored[scored.split.eq("oot")],config))
    variances=[]
    for split in ("train","validation","oot"):
        part=scored[scored.split.eq(split)]
        variances.append({"split":split,**episode_variance(part.score,part.episode_id),
                          **contribution_variance(part.individual_points,part.market_points)})
    reports.table("score_variance",variances)
    # Stability diagnostics use frozen bins and the same C; never feed into final selection.
    stability=[]
    for year,part in train.groupby(train.t0.dt.year):
        if part.BAD.nunique()<2:
            continue
        yearly_model,converged=fit_logistic(train_woe.loc[part.index,selected].to_numpy(),part.BAD.astype(int),float(chosen.C),config)
        for feature,beta,reference in zip(selected,yearly_model.coef_[0],coefficients):
            stability.append({"year":int(year),"feature":feature,"coefficient":beta,
                              "full_train_coefficient":reference,"sign_agrees":np.sign(beta)==np.sign(reference),"converged":converged})
    reports.table("coefficient_stability",stability)
    announce("[7/8] Human review items and reproducibility manifest")
    oot_metrics=metric_frame[(metric_frame.split=="oot")&(metric_frame.model=="final")].iloc[0].to_dict()
    review=[
        {"item":"관측 종료일", "status":"required", "detail":f"{config.observation_end}은 폐업 이력 완전성 확인 전 가정. 실제 수집범위 확인 필요."},
        {"item":"추가 변수 시점", "status":"required", "detail":"지하철·서비스인구·주민인구는 기준일/공표일/조인 근거 확인 전 제외. extra_feature_availability로 검증 후 추가."},
        {"item":"구간 경제적 설명", "status":"required", "detail":"자동 단조 구간의 업력·경쟁도·매출 해석, 고IV·음수 계수와 작은 결측 구간을 검토."},
        {"item":"등급 정책", "status":"required", "detail":"Validation 실제 폐업률 기반 후보 등급. 지원 규모·위험수용도와 정책 사용 기준은 사람이 확정."},
        {"item":"적용범위", "status":"required", "detail":"합성 매출 위기 노출자의 폐업모형이며 신용채무 불이행 확률이 아님. 실제 매출 확보 후 재검증."},
        {"item":"표본 누락", "status":"required", "detail":"enrichment 데이터의 원본 대비 누락 사유 확인. 중복은 제거했으나 입력에 없는 사업자를 복원하지 않음."}]
    for feature in stats.index[stats.high_iv_review]:
        review.append({"item":feature,"status":"high_iv","detail":f"Train IV={stats.loc[feature,'iv']:.4f}: 누수/구간 편중 확인"})
    for feature in coefficient_table.loc[coefficient_table.negative_sign_review,"feature"]:
        review.append({"item":feature,"status":"negative_coefficient","detail":"BAD WOE와 조건부 계수 방향이 반대: 다중공선성/경제적 설명 검토"})
    for message in grade_definition["warnings"]:
        review.append({"item":"등급 수","status":"review","detail":message})
    for split,part in grade_table.groupby("split"):
        rates=part.actual_bad_rate.dropna().to_numpy()
        if np.any(np.diff(rates)<=0):
            review.append({"item":f"{split} 등급","status":"nonmonotone","detail":"등급별 실제 폐업률 역전 또는 동률. OOT를 보고 경계를 재조정하지 않음."})
    for row in psi_frame[(psi_frame.comparison=="oot")&psi_frame.review_flag&psi_frame.in_final_model].to_dict("records"):
        review.append({"item":row["feature"],"status":"oot_psi_high","detail":f"OOT PSI={row['psi']:.4f}; 임계값 .25는 검토 기준, 자동 적합 판정 아님"})
    reports.table("human_review",review)
    announce("Saving intermediate model charts to figures/")
    figures = save_figures(reports.directory)
    source_files={p.name:fingerprint(p) for p in Path(__file__).parent.glob("*.py")}
    summary={"run_id":run_id,"horizon_months":config.horizon_months,"status":"development_candidate_requires_review",
             "data":audit,"split_counts":frame.split.value_counts().to_dict(),"selected_features":selected,
             "selected_C":float(chosen.C),"calibration":calibration,"grades":grade_definition["grades"],
             "oot_metrics":oot_metrics,"assumptions":["observation_end is assumed complete","synthetic market sales","unverified enrichment excluded", "validation statistics used for tuning are development estimates"],
             "source":source_hash,"metadata":fingerprint(config.metadata),"code":source_files,
             "figures":figures,
             "versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
             "elapsed_seconds":time.perf_counter()-started,"model_json_replay_verified":True,
             "score_contributions_verified":True,"report_dir":str(reports.directory.resolve()),"model_dir":str(model_dir.resolve())}
    reports.object("run_summary",summary)
    save_json(model_dir/"manifest.json",summary)
    result={"report_dir":str(reports.directory.resolve()),"model_dir":str(model_dir.resolve()),"summary":summary}
    save_json(Path(config.report_root)/"latest_run.json",result)
    announce(f"[8/8] Complete: {reports.directory}")
    announce(json.dumps({"selected_features":selected,"oot":oot_metrics},ensure_ascii=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,help="JSON overrides for ScorecardConfig")
    parser.add_argument("--horizon",type=int,choices=[12,18,24])
    arguments=parser.parse_args()
    options=json.loads(arguments.config.read_text(encoding="utf-8-sig")) if arguments.config else {}
    if arguments.horizon:
        options["horizon_months"]=arguments.horizon
    run_pipeline(ScorecardConfig(**options))


if __name__=="__main__":
    main()
