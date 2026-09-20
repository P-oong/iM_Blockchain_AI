"""Explicit, serializable development settings. Dates are half-open boundaries."""
from dataclasses import asdict, dataclass, field


@dataclass
class ScorecardConfig:
    source: str = "data/synthetic/dmdrlt4.csv"
    metadata: str = "data/processed/labeling/labeling.metadata.json"
    report_root: str = "data/processed/scorecard"
    model_root: str = "src/models/artifacts"
    horizon_months: int = 12
    observation_end: str = "2025-12-31"
    valid_start: str = "2022-01-01"
    valid_end: str = "2023-01-01"
    oot_start: str = "2024-01-01"
    oot_end: str = "2025-01-01"
    include_district: bool = True
    # Extras require a per-row availability date column, not an unsupported assertion.
    extra_feature_availability: dict[str, str] = field(default_factory=dict)
    excluded_features: list[str] = field(default_factory=list)
    missing_limit: float = .95
    dominant_limit: float = .95
    iv_min: float = .02
    iv_review: float = .5
    correlation_limit: float = .7
    fine_bins: int = 10
    max_bins: int = 5
    min_bin_fraction: float = .05
    min_bin_bad: int = 20
    min_bin_good: int = 20
    smoothing: float = .5
    c_grid: list[float] = field(default_factory=lambda: [.01, .05, .1, .3, .5, 1., 2.])
    auc_tolerance: float = .005
    max_iter: int = 3000
    random_seed: int = 123
    base_score: float = 600.
    base_odds: float = 20.
    pdo: float = 50.
    max_grades: int = 5
    min_grade_fraction: float = .05
    min_grade_bad: int = 10
    cluster_bootstrap_repeats: int = 200

    def to_dict(self):
        return asdict(self)

    def validate(self):
        import pandas as pd
        if self.horizon_months not in (12, 18, 24):
            raise ValueError("horizon_months must be 12, 18 or 24")
        dates = [pd.Timestamp(x) for x in (self.valid_start, self.valid_end, self.oot_start, self.oot_end)]
        if not dates[0] < dates[1] <= dates[2] < dates[3]:
            raise ValueError("Require valid_start < valid_end <= oot_start < oot_end")
        if not self.c_grid or any(c <= 0 for c in self.c_grid):
            raise ValueError("c_grid must contain positive values")
        if not 0 <= self.iv_min < self.iv_review:
            raise ValueError("Require 0 <= iv_min < iv_review")
        for name in ("missing_limit", "dominant_limit", "correlation_limit", "min_bin_fraction", "min_grade_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if self.base_odds <= 0 or self.pdo <= 0 or self.cluster_bootstrap_repeats < 0:
            raise ValueError("Invalid score scaling/bootstrap settings")
