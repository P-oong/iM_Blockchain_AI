"""Save reproducible scorecard diagnostic images from the exported CSV reports.

No fitting, bin selection or grade adjustment happens here. Every image cites
its source CSV in figure_manifest.csv/json. All predictions mean BAD=1 closure.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter
import numpy as np
import pandas as pd

from .evaluation import calibration_table, classification_metrics


SPLITS = ("train", "validation", "oot")
COLORS = {"train": "#337AB7", "validation": "#E69F00", "oot": "#009E73"}


def _filename(value):
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:48] or "feature"
    return slug + "_" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]


def _font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic",
                                  "AppleGothic", "DejaVu Sans") if name in available), "sans-serif")


def _percent(axis, which="y"):
    (axis.yaxis if which == "y" else axis.xaxis).set_major_formatter(PercentFormatter(1))


def _curves(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    tp = np.cumsum(y[order])[ends]
    fp = ends + 1 - tp
    tpr = np.r_[0., tp / y.sum()]
    fpr = np.r_[0., fp / (len(y) - y.sum())]
    precision = np.r_[1., tp / (ends + 1)]
    selected = np.r_[0., (ends + 1) / len(y)]
    return fpr, tpr, precision, selected


def save_figures(report_dir) -> dict:
    """Generate PNG diagnostics and CSV/JSON image manifests; return locations.

    Missing/empty optional tables are skipped and listed in the returned
    summary. Rendering errors are raised so runs cannot claim images exist
    when generation failed. Figures are closed after saving.
    """
    root = Path(report_dir)
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    manifest, skipped, tables = [], [], {}

    def read(name):
        if name not in tables:
            try:
                tables[name] = pd.read_csv(root / f"{name}.csv", encoding="utf-8-sig")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                tables[name] = pd.DataFrame()
                skipped.append(name)
        return tables[name]

    def save(fig, name, description, sources):
        path = directory / f"{name}.png"
        try:
            fig.tight_layout()
            fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
        finally:
            plt.close(fig)
        manifest.append({"figure": name, "path": str(path.relative_to(root)),
                         "description": description, "source_csv": ";".join(f"{x}.csv" for x in sources),
                         "format": "PNG", "dpi": 160})

    style = {"font.family": _font(), "axes.unicode_minus": False, "font.size": 9,
             "mathtext.fontset": "dejavusans", "mathtext.default": "rm",
             "axes.titlesize": 11, "axes.labelsize": 9, "axes.spines.top": False,
             "axes.spines.right": False, "figure.facecolor": "white"}
    with plt.rc_context(style):
        split = read("split_summary")
        if not split.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].bar(split.split, split["count"], color="#8FAFC7")
            axes[0].set(ylabel="Businesses", title="Time split and embargo populations")
            for i, count in enumerate(split["count"]):
                axes[0].text(i, count, f"{int(count):,}", ha="center", va="bottom", fontsize=8)
            observed = split[split.split.isin(SPLITS)]
            axes[1].bar(observed.split, observed.bad_rate, color=[COLORS[s] for s in observed.split])
            axes[1].set(ylabel="Observed closure rate (BAD=1)", title="Outcome distribution by split")
            _percent(axes[1])
            save(fig, "01_split_target", "표본 분리 및 분리별 폐업률; embargo는 학습/검증에서 제외", ["split_summary"])

        screening = read("feature_screening")
        if not screening.empty:
            part = screening.sort_values("missing_rate")
            fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(part) * .25)))
            axes[0].barh(part.feature, part.missing_rate, color="#337AB7")
            axes[0].set(xlabel="Missing share", title="Training missingness")
            axes[1].barh(part.feature, part.dominant_share, color="#8A9A5B")
            axes[1].set(xlabel="Most frequent value share", title="Training dominance")
            _percent(axes[0], "x"); _percent(axes[1], "x")
            save(fig, "02_feature_quality", "Train 변수별 결측률 및 최빈값 점유율", ["feature_screening"])

        iv = read("woe_summary")
        if not iv.empty:
            part = iv.sort_values("iv")
            fig, ax = plt.subplots(figsize=(11, max(5, len(part) * .26)))
            ax.barh(part.feature, part.iv, color=np.where(part.iv >= .5, "#C44E52", "#337AB7"))
            ax.axvline(.02, linestyle="--", color="#777777", label="IV 0.02 reference")
            ax.set(xlabel="Information value (IV)", title="Training-only IV — high IV requires review")
            ax.legend(loc="lower right")
            save(fig, "03_training_iv", "Train WOE 정보가치; 높은 IV는 누수·편중 검토 대상", ["woe_summary"])

        selection = read("selection_log")
        if not selection.empty:
            stages = [name for name in ("quality", "iv", "correlation", "l1") if name in set(selection.stage)]
            counts = [int(selection.loc[selection.stage.eq(stage), "decision"].isin(["keep", "pass"]).sum()) for stage in stages]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(stages, counts, color="#337AB7")
            for i, count in enumerate(counts):
                ax.text(i, count, str(count), va="bottom", ha="center")
            ax.set(ylabel="Retained variables", title="Automated variable-selection stages")
            save(fig, "04_selection_counts", "품질→IV→상관성→L1 단계별 잔존 변수 수", ["selection_log"])

        fine, coarse = read("woe_fine"), read("woe_coarse")
        if not fine.empty and not coarse.empty:
            for feature in fine.feature.drop_duplicates():
                fig, axes = plt.subplots(2, 2, figsize=(13, 8))
                for column, (name, table) in enumerate((("Fine", fine), ("Coarse", coarse))):
                    part = table[table.feature.eq(feature)].reset_index(drop=True)
                    x = np.arange(len(part))
                    labels = part.bin_id.astype(str)
                    axes[0, column].bar(x, part["count"], color="#BDD7E7", label="Count")
                    axes[0, column].set(ylabel="Training businesses", title=f"{name} bin count and BAD rate")
                    twin = axes[0, column].twinx()
                    twin.plot(x, part.bad_rate, color="#C44E52", marker="o", label="BAD rate")
                    twin.set_ylabel("Observed closure rate"); _percent(twin)
                    axes[1, column].bar(x, part.woe, color=np.where(part.woe >= 0, "#C44E52", "#337AB7"))
                    axes[1, column].axhline(0, color="#444444", linewidth=.7)
                    axes[1, column].set(ylabel="WOE = ln(BAD share / GOOD share)", title=f"{name} WOE")
                    for row in (0, 1):
                        axes[row, column].set_xticks(x, labels, rotation=55, ha="right")
                fig.suptitle(f"{feature} — train-only binning", fontsize=13)
                save(fig, "05_woe_" + _filename(feature), f"{feature}: Fine/Coarse 구간 인원·폐업률·WOE (정확한 구간 정의는 CSV 참조)", ["woe_fine", "woe_coarse"])

        correlation = read("correlation_matrix")
        if not correlation.empty:
            values = correlation.set_index("feature")
            size = max(7, len(values) * .4)
            fig, ax = plt.subplots(figsize=(size, size))
            picture = ax.imshow(values.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(np.arange(len(values.columns)), values.columns, rotation=70, ha="right")
            ax.set_yticks(np.arange(len(values)), values.index)
            ax.set_title("Training WOE Spearman correlation")
            fig.colorbar(picture, ax=ax, shrink=.75, label="Spearman correlation")
            save(fig, "06_correlation", "Train WOE 변수 Spearman 상관행렬", ["correlation_matrix"])

        tuning = read("c_tuning")
        if not tuning.empty:
            tuning = tuning.sort_values("C")
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, column, title in zip(axes, ("roc_auc", "brier", "n_features"), ("ROC-AUC (higher is better)", "Brier (lower is better)", "Active variables")):
                ax.plot(tuning.C, tuning[column], marker="o", color="#337AB7")
                selected = tuning[tuning.selected.astype(str).str.lower().eq("true")]
                if not selected.empty:
                    ax.scatter(selected.C, selected[column], marker="*", s=130, color="#C44E52", zorder=3, label="Selected C")
                    ax.legend()
                ax.set(xlabel="C (inverse regularization)", title=title, xscale="log")
                # Plain decimal tick labels avoid Korean-font math minus gaps.
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            fig.suptitle("Validation tuning — OOT excluded", fontsize=13)
            save(fig, "07_c_tuning", "Validation C 후보별 AUC·Brier·활성 변수 수와 선택 결과", ["c_tuning"])

        coefficients = read("coefficients")
        if not coefficients.empty:
            part = coefficients.sort_values("calibrated_coefficient")
            fig, ax = plt.subplots(figsize=(10, max(4, len(part) * .4)))
            y = np.arange(len(part))
            ax.barh(y - .17, part.coefficient, height=.32, label="Raw logistic")
            ax.barh(y + .17, part.calibrated_coefficient, height=.32, label="Final calibrated")
            ax.set_yticks(y, part.feature); ax.axvline(0, color="#777777", linewidth=.8)
            ax.set(xlabel="Log-odds coefficient per WOE unit", title="Final selected coefficients — target BAD=1")
            ax.legend()
            save(fig, "08_coefficients", "선택 변수의 로지스틱 원계수 및 확률 보정 후 계수", ["coefficients"])

        stability = read("coefficient_stability")
        if not stability.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            for feature, part in stability.groupby("feature", sort=False):
                part = part.sort_values("year")
                ax.plot(part.year, part.coefficient, marker="o", label=feature)
            ax.axhline(0, color="#777777", linewidth=.8)
            ax.set(xlabel="Training year", ylabel="WOE coefficient", title="Within-training coefficient stability (diagnostic refits)")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            save(fig, "09_coefficient_stability", "고정 구간과 C를 사용한 Train 연도별 계수 진단", ["coefficient_stability"])

        scored = read("scored_businesses")
        if not scored.empty:
            for split_name in SPLITS:
                part = scored[scored.split.eq(split_name)].dropna(subset=["BAD", "p_bad", "p_bad_raw"])
                if part.empty:
                    continue
                fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
                if part.BAD.nunique() == 2:
                    for kind, column, color, linestyle in (("Raw", "p_bad_raw", "#999999", "--"), ("Final", "p_bad", COLORS[split_name], "-")):
                        fpr, tpr, precision, selected = _curves(part.BAD, part[column])
                        metrics = classification_metrics(part.BAD, part[column])
                        axes[0].plot(fpr, tpr, color=color, linestyle=linestyle, label=f"{kind} AUC={metrics['roc_auc']:.3f}")
                        axes[1].plot(selected, tpr - fpr, color=color, linestyle=linestyle, label=f"{kind} |KS|max={metrics['ks']:.3f}")
                        axes[2].plot(tpr, precision, color=color, linestyle=linestyle, label=f"{kind} PR-AUC={metrics['pr_auc']:.3f}; AP={metrics['average_precision']:.3f}")
                    axes[0].plot([0, 1], [0, 1], ":", color="#777777")
                    axes[2].axhline(part.BAD.mean(), color="#777777", linestyle=":", label="Observed BAD rate")
                    for ax in axes:
                        ax.legend(fontsize=8, loc="best")
                else:
                    for ax in axes:
                        ax.text(.5, .5, "One outcome class: discrimination undefined", transform=ax.transAxes, ha="center", wrap=True)
                axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC", xlim=(0, 1), ylim=(0, 1))
                axes[1].set(xlabel="Share selected, highest P(BAD) first", ylabel="Cumulative BAD share - GOOD share", title="KS separation", xlim=(0, 1))
                axes[1].axhline(0, color="#777777", linewidth=.8)
                axes[2].set(xlabel="Recall", ylabel="Precision", title="Precision–recall", xlim=(0, 1), ylim=(0, 1.02))
                fig.suptitle(f"{split_name.upper()} — BAD=1 closure; frozen predictions", fontsize=13)
                save(fig, "10_discrimination_" + split_name, f"{split_name}: 보정 전/후 ROC·KS·PR 곡선", ["scored_businesses"])

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for ax, split_name in zip(axes, SPLITS):
                part = scored[scored.split.eq(split_name)].dropna(subset=["BAD", "p_bad", "p_bad_raw"])
                for kind, column, color, linestyle in (("Raw", "p_bad_raw", "#888888", "--"), ("Final", "p_bad", COLORS[split_name], "-")):
                    if part.empty:
                        continue
                    curve = calibration_table(part.BAD, part[column])
                    ax.plot(curve.mean_p_bad, curve.actual_bad_rate, marker="o", linestyle=linestyle, color=color, label=kind)
                ax.plot([0, 1], [0, 1], ":", color="#777777", label="Ideal")
                ax.set(xlabel="Mean predicted P(BAD)", ylabel="Observed closure rate", title=split_name.upper(), xlim=(0, 1), ylim=(0, 1))
                _percent(ax); _percent(ax, "x"); ax.legend()
            fig.suptitle("Probability calibration — quantile bins, ties preserved", fontsize=13)
            save(fig, "11_calibration", "Train/Validation/OOT 보정 전후 확률 및 실제 폐업률", ["scored_businesses"])

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            included = scored[scored.split.isin(SPLITS)]
            score_min, score_max = included.score.min(), included.score.max()
            if score_min == score_max:
                score_min, score_max = score_min - 1, score_max + 1
            edges = np.linspace(score_min, score_max, 41)
            for split_name in SPLITS:
                part = included[included.split.eq(split_name)]
                if not part.empty:
                    axes[0].hist(part.score, bins=edges, density=True, histtype="step", linewidth=1.8, label=split_name, color=COLORS[split_name])
            oot = included[included.split.eq("oot")]
            for label, color, title in ((0, "#337AB7", "Survived BAD=0"), (1, "#C44E52", "Closed BAD=1")):
                values = oot.loc[oot.BAD.eq(label), "score"]
                if len(values):
                    axes[1].hist(values, bins=edges, density=True, histtype="step", linewidth=1.8, label=title, color=color)
            for ax, title in zip(axes, ("Score distributions by time split", "OOT score by observed outcome")):
                ax.set(xlabel="Score (higher = lower closure risk)", ylabel="Density", title=title); ax.legend()
            save(fig, "12_score_distributions", "분리별 평점 분포와 OOT 생존/폐업별 평점 분포", ["scored_businesses"])

        grade = read("grade_performance")
        if not grade.empty:
            fig, axes = plt.subplots(3, 2, figsize=(12, 11))
            for row, split_name in enumerate(SPLITS):
                part = grade[grade.split.eq(split_name)]
                axes[row, 0].bar(part.grade, part.share, color=COLORS[split_name])
                axes[row, 0].set(title=f"{split_name.upper()} grade population", ylabel="Business share")
                axes[row, 1].plot(part.grade, part.actual_bad_rate, "o-", color="#C44E52", label="Observed BAD rate")
                axes[row, 1].plot(part.grade, part.mean_p_bad, "s--", color="#337AB7", label="Mean predicted P(BAD)")
                axes[row, 1].set(title=f"{split_name.upper()} grade risk", ylabel="Closure probability")
                for ax in axes[row]:
                    _percent(ax)
                axes[row, 1].legend(fontsize=8)
            fig.suptitle("Validation-frozen grades — R1 lowest predicted risk", fontsize=13)
            save(fig, "13_grades", "고정 등급의 표본 점유율 및 실제/예측 폐업률; OOT 역전도 그대로 표시", ["grade_performance"])

        psi = read("psi_summary")
        if not psi.empty:
            score_psi = psi[psi.feature.eq("TOTAL_SCORE")]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(score_psi.comparison, score_psi.psi, color="#337AB7")
            ax.axhline(.25, color="#C44E52", linestyle="--", label="0.25 review reference")
            ax.tick_params(axis="x", rotation=45)
            ax.set(ylabel="PSI", title="Total-score PSI — fixed training score bins"); ax.legend()
            save(fig, "14_score_psi", "Train 대비 전체 평점 PSI; 고정 Train 구간 사용", ["psi_summary"])
            comparisons = psi[psi.comparison.isin(["validation", "oot"]) & psi.feature.ne("TOTAL_SCORE")]
            if not comparisons.empty:
                pivot = comparisons.pivot(index="feature", columns="comparison", values="psi")
                pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
                fig, ax = plt.subplots(figsize=(11, max(5, len(pivot) * .32)))
                y = np.arange(len(pivot))
                for i, column in enumerate(pivot.columns):
                    ax.barh(y + (i - .5) * .35, pivot[column], height=.33, label=column, color=COLORS[column])
                ax.set_yticks(y, pivot.index); ax.axvline(.25, color="#C44E52", linestyle="--")
                ax.set(xlabel="PSI", title="Variable PSI — validation and OOT vs training"); ax.legend()
                save(fig, "15_variable_psi", "변수별 Validation/OOT PSI; 관측하지 못한 범주도 포함", ["psi_summary"])

        years = read("metrics_by_year")
        if not years.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for split_name in SPLITS:
                part = years[years.split.eq(split_name)].sort_values("year")
                for ax, column in zip(axes, ("roc_auc", "brier")):
                    ax.plot(part.year, part[column], marker="o", label=split_name, color=COLORS[split_name])
            for ax, title in zip(axes, ("Annual ROC-AUC", "Annual Brier score")):
                ax.set(xlabel="Crisis year", title=title); ax.legend()
            save(fig, "16_annual_metrics", "고정 모형의 기준연도별 AUC 및 Brier", ["metrics_by_year"])

        variance = read("score_variance")
        if not variance.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            within = variance.within_episode_share.fillna(0)
            between = variance.between_episode_share.fillna(0)
            axes[0].bar(variance.split, within, color="#337AB7", label="Within episode")
            axes[0].bar(variance.split, between, bottom=within, color="#E69F00", label="Between episodes")
            axes[0].set(ylabel="Share of score variance", title="Within / between market-crisis episodes"); _percent(axes[0]); axes[0].legend()
            x = np.arange(len(variance))
            axes[1].bar(x - .18, variance.individual_allocated_share, width=.35, color="#337AB7", label="Individual + covariance allocation")
            axes[1].bar(x + .18, variance.market_allocated_share, width=.35, color="#E69F00", label="Market + covariance allocation")
            axes[1].set_xticks(x, variance.split); axes[1].axhline(0, color="#777777", linewidth=.8)
            axes[1].set(ylabel="Allocated variance share (can be negative)", title="Additive score contribution — not causal importance")
            _percent(axes[1]); axes[1].legend(fontsize=7)
            save(fig, "17_score_variance", "위기군 내/간 점수 분산 및 공분산을 배분한 개인/상권 기여분; 인과적 중요도 아님", ["score_variance"])

        annual_bins, annual_psi = read("training_year_bins"), read("training_year_psi")
        if not annual_bins.empty and not annual_psi.empty:
            for feature, values in annual_bins.groupby("feature", sort=False):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
                for bin_id, part in values.groupby("bin_id", sort=False):
                    part = part.sort_values("year")
                    axes[0].plot(part.year, part.bad_rate, marker="o", label=bin_id)
                axes[0].set(xlabel="Training year", ylabel="Observed closure rate", title="Fixed-bin annual BAD rates")
                _percent(axes[0]); axes[0].legend(fontsize=7)
                part = annual_psi[annual_psi.feature.eq(feature)].sort_values("year")
                axes[1].bar(part.year, part.psi, color="#337AB7")
                axes[1].axhline(.25, color="#C44E52", linestyle="--")
                axes[1].set(xlabel="Training year", ylabel="PSI vs all training rows", title="Within-training population stability")
                fig.suptitle(feature, fontsize=13)
                save(fig, "18_time_stability_" + _filename(feature), f"{feature}: 고정 구간별 Train 연도 폐업률과 연도 PSI", ["training_year_bins", "training_year_psi"])

    manifest_frame = pd.DataFrame(manifest, columns=["figure", "path", "description", "source_csv", "format", "dpi"])
    manifest_csv, manifest_json = root / "figure_manifest.csv", root / "figure_manifest.json"
    manifest_frame.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"count": len(manifest), "directory": str(directory.resolve()),
            "manifest_csv": str(manifest_csv.resolve()), "manifest_json": str(manifest_json.resolve()),
            "skipped_missing_tables": skipped, "figures": manifest}
