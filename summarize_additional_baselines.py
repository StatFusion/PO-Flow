"""Aggregate and audit final TARNet and DragonNet runs."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "additional_baselines"
FORMAL = re.compile(r".+_(TARNet|DragonNet)_seed(42|43|44|45|46)_metrics\.csv$")
METHODS = {"TARNet", "DragonNet"}
SEEDS = {42, 43, 44, 45, 46}
DATASETS = {
    "ACIC2016", "ACIC2018", "IHDP", "IBM",
    "IHDP_Vector_d2", "IHDP_Vector_d5", "IHDP_Vector_d10",
    "ACIC2016_Vector_d2", "ACIC2016_Vector_d5", "ACIC2016_Vector_d10",
    "MorphoMNIST_Intensity_d784",
}
METRICS = ["counterfactual", "pehe", "po", "distribution_w1", "mean_w1"]


def _poflow_scalar():
    frame = pd.read_csv(ROOT / "results" / "final_ipw" / "final_ipw_summary.csv")
    frame = frame[frame.dataset.isin({"ACIC2016", "ACIC2018", "IHDP", "IBM"})].copy()
    frame["method"] = "PO-Flow+IPW"
    return frame[["dataset", "method"] + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]]


def _poflow_vector():
    frame = pd.read_csv(ROOT / "results" / "optimization_v3" / "final_summary.csv")
    frame["method"] = "PO-Flow+IPW"
    return frame[["dataset", "method"] + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]]


def _existing_vector_baselines():
    frame = pd.read_csv(ROOT / "results" / "multidim_baselines" / "multidim_baseline_summary.csv")
    rename = {}
    source_names = {
        "counterfactual_NRMSE": "counterfactual",
        "vector_PEHE": "pehe",
        "PO_NRMSE": "po",
        "sliced_W1_distance": "distribution_w1",
        "sliced_W1_mean_distance": "mean_w1",
    }
    for source, target in source_names.items():
        for stat in ("mean", "std"):
            rename[f"{source}_{stat}"] = f"{target}_{stat}"
    frame = frame.rename(columns=rename)
    return frame[["dataset", "method"] + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]]


def build_journal_comparisons(summary):
    value_columns = [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    scalar_new = summary[summary.dataset.isin({"ACIC2016", "ACIC2018", "IHDP", "IBM"})]
    scalar = pd.concat(
        [_poflow_scalar(), scalar_new[["dataset", "method"] + value_columns]],
        ignore_index=True,
    ).sort_values(["dataset", "method"])
    scalar.to_csv(RESULTS / "scalar_journal_comparison.csv", index=False)

    vector_new = summary[~summary.dataset.isin({"ACIC2016", "ACIC2018", "IHDP", "IBM"})]
    vector = pd.concat(
        [_poflow_vector(), _existing_vector_baselines(), vector_new[["dataset", "method"] + value_columns]],
        ignore_index=True,
    ).sort_values(["dataset", "method"])
    vector.to_csv(RESULTS / "vector_journal_comparison.csv", index=False)

    winners = []
    for dataset, block in vector.groupby("dataset"):
        for metric in METRICS:
            column = f"{metric}_mean"
            applicable = block[np.isfinite(block[column])]
            minimum = applicable[column].min()
            for _, row in applicable[np.isclose(applicable[column], minimum, rtol=0.0, atol=1e-12)].iterrows():
                winners.append({"dataset": dataset, "metric": metric, "method": row.method, "value": row[column]})
    winners = pd.DataFrame(winners).sort_values(["dataset", "metric", "method"])
    winners.to_csv(RESULTS / "vector_journal_winners.csv", index=False)
    counts = winners.groupby("method").size().rename("winner_count").reset_index().sort_values("winner_count", ascending=False)
    counts.to_csv(RESULTS / "vector_journal_winner_counts.csv", index=False)
    return scalar, vector, winners, counts


def main():
    files = sorted(path for path in RESULTS.glob("*_metrics.csv") if FORMAL.fullmatch(path.name))
    if not files:
        raise SystemExit("no formal TARNet/DragonNet metrics found")
    rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    rows = rows.sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
    if rows[["dataset", "method", "seed"]].duplicated().any():
        raise ValueError("duplicate formal run keys")
    if len(rows) == 110:
        keys = set(map(tuple, rows[["dataset", "method", "seed"]].itertuples(index=False, name=None)))
        expected = {(dataset, method, seed) for dataset in DATASETS for method in METHODS for seed in SEEDS}
        if keys != expected:
            raise ValueError(f"formal key mismatch: missing={sorted(expected-keys)} extra={sorted(keys-expected)}")
    point_values = rows[["counterfactual", "pehe", "po", "mean_w1"]].to_numpy(float)
    if not np.isfinite(point_values).all() or (point_values < 0).any():
        raise ValueError("nonfinite or negative point metric")
    if rows["distribution_w1"].notna().any() or rows["distributional_metric_applicable"].astype(bool).any():
        raise ValueError("point baselines must not report distributional W1")
    rows.to_csv(RESULTS / "additional_baseline_all_seed_metrics.csv", index=False)
    summary = rows.groupby(["dataset", "method"])[METRICS + ["wall_seconds", "parameter_count"]].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS / "additional_baseline_summary.csv", index=False)
    scalar, vector, winners, counts = build_journal_comparisons(summary)
    print(summary.to_string(index=False))
    print("\nHigh-dimensional winner counts:")
    print(counts.to_string(index=False))
    print(f"\nAudited {len(rows)}/110 expected runs; wall time sum={rows.wall_seconds.sum():.3f}s")


if __name__ == "__main__":
    main()
