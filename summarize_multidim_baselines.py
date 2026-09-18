"""Aggregate formal multidimensional CFR/GANITE/CEVAE runs."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "multidim_baselines"
FLOW_RESULTS = ROOT / "results" / "multidim"
FORMAL = re.compile(r".+_(CFR|GANITE|CEVAE)_seed(42|43|44|45|46)_metrics\.csv$")
METRICS = [
    "counterfactual_NRMSE",
    "vector_PEHE",
    "PO_NRMSE",
    "sliced_W1_distance",
    "sliced_W1_mean_distance",
]


def main():
    files = sorted(path for path in RESULTS.glob("*_metrics.csv") if FORMAL.fullmatch(path.name))
    if not files:
        raise SystemExit("no formal multidimensional baseline metrics found")
    rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    rows = rows.sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
    if rows[["dataset", "method", "seed"]].duplicated().any():
        raise ValueError("duplicate formal run keys")
    rows.to_csv(RESULTS / "multidim_baseline_all_seed_metrics.csv", index=False)

    summary = rows.groupby(["dataset", "method"])[METRICS].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS / "multidim_baseline_summary.csv", index=False)

    flow = pd.read_csv(FLOW_RESULTS / "multidim_summary.csv")
    flow["method"] = flow["method"].map({"Baseline": "PO-Flow Base", "IPW": "PO-Flow IPW"})
    flow_columns = ["dataset", "method"]
    for metric in METRICS:
        flow_columns.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_count"))
    combined = pd.concat([summary, flow[flow_columns]], ignore_index=True, sort=False)
    combined = combined.sort_values(["dataset", "method"]).reset_index(drop=True)
    combined.to_csv(RESULTS / "multidim_all_methods_summary.csv", index=False)

    reference = flow[flow.method.eq("PO-Flow IPW")].set_index("dataset")
    comparisons = []
    for _, row in summary.iterrows():
        ref = reference.loc[row.dataset]
        item = {"dataset": row.dataset, "method": row.method, "reference": "PO-Flow IPW"}
        for metric in METRICS:
            value = row[f"{metric}_mean"]
            reference_value = ref[f"{metric}_mean"]
            item[f"{metric}_relative_percent"] = (
                100.0 * (value / reference_value - 1.0)
                if np.isfinite(value) and np.isfinite(reference_value) else np.nan
            )
        comparisons.append(item)
    pd.DataFrame(comparisons).to_csv(RESULTS / "multidim_baseline_vs_poflow_ipw.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
