"""Verify that this checkout matches the frozen final PO-Flow release."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CORE_HASHES = {
    "main.py": "02a62af4a9cba6057506cbfaa29ca184473299458d10045c4c535d2677c5546b",
    "main_IPW.py": "74825733c6ba1b5a5429afc1d28ff2787ea0cd493af36e2db8103718ea12fd16",
    "main_multidim.py": "83888ee312a46b7042a8d65b17d9dffeef0c91868b64da9fc1163266bcff1095",
    "ipw.py": "007cac8e74800fc09ac6c5712110fdb8b92ef2d4d8a4c7fc51953a5ef44e1766",
    "run_tuned_ipw.py": "457948fb42f9ec81a16d4b194906d159f200938bc2cbc6cdf5a69b58f210a1e8",
}

CONFIG_HASHES = {
    "configs/scalar/ACIC2016.yaml": "2ede0fb6456c998f5c40c781a09f87ab422dc49cd2796520e469eb4e31c625cf",
    "configs/scalar/ACIC2018.yaml": "1bae40011de5a3091c40db7e48f5e131443c538f135cae9e8c2bc92ca7d13920",
    "configs/scalar/IBM.yaml": "83a76851692ee205da4823ea48e3d15877083228480a98a26d8f2446fb52ce65",
    "configs/scalar/IHDP.yaml": "2c07b8f53b2d9f263d54b6ee30055606fee10d24854830229ade9569831079fe",
    "configs/vector/ACIC2016_Vector_d10.yaml": "8ec0102bc245b94185666eef4fd7a89cdf4632cd166d2282774202eef84f7406",
    "configs/vector/ACIC2016_Vector_d2.yaml": "135a55cda696f83285af4d37372186e4eae39027df44c76a9e01864750951505",
    "configs/vector/ACIC2016_Vector_d5.yaml": "217e1b0b4785df7898d70c1d7d6c0731da44a23b512c1ff9e94c4f04a363fa77",
    "configs/vector/IHDP_Vector_d10.yaml": "58749089b5d0f674b7e3b33a8e9ac01325b3efb2946662716b12bae13e11d337",
    "configs/vector/IHDP_Vector_d2.yaml": "5ee10b4d119427c526d3c5e052e5f2c6b200ef75b192d843c405a5230e1854e8",
    "configs/vector/IHDP_Vector_d5.yaml": "f9bbe3b75f4bfb71cb33ab4b6247e46186cf1b908bd77bd229165fe868107bb4",
    "configs/vector/MorphoMNIST_Intensity_d784.yaml": "58742b13606c15118c7b9623c9faaa53baf674e65e07a265520c6ecfecfb6382",
}

DATA_HASHES = {
    "multidim_benchmarks/processed/acic2016_vector_d10.npz": "49848e06f84adef0f4882399c5d95d5a56b15b211935a48a944ad2eb392ca07a",
    "multidim_benchmarks/processed/acic2016_vector_d2.npz": "e7ce445ad7c667e244531d1ad8aace71cb3a5625744c1bf22f1bc74debee8ff4",
    "multidim_benchmarks/processed/acic2016_vector_d5.npz": "692d43743904ba265d3ec235c951109ef07547ee70b8582021d2b7038ac8e828",
    "multidim_benchmarks/processed/ihdp_vector_d10.npz": "8eb4e11020a5e8eec5a7c95a93893e5a01c1611bc99d5f4b3f13ee6f1403b8be",
    "multidim_benchmarks/processed/ihdp_vector_d2.npz": "03d0de47c97b2af56d8639b42365369d76eee73ec205c2977881e042549bed90",
    "multidim_benchmarks/processed/ihdp_vector_d5.npz": "fa062b6279dd261c783397ed4cc71a6a5b7b6848dabad4556443fa459cb6b525",
    "multidim_benchmarks/processed/morphomnist_intensity_d784.npz": "2f00e22b6267e361fcdf3552f471999dffeb7f75aa02ed2a67acf6d8dc28ef0d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(expected):
    failures = []
    for relative, expected_digest in expected.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected_digest:
            failures.append(f"hash mismatch: {relative}\n  expected {expected_digest}\n  actual   {actual}")
    return failures


def verify_results():
    path = ROOT / "reproducibility" / "final_results.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_datasets = {Path(path).stem for path in CONFIG_HASHES}
    observed_datasets = {row["dataset"] for row in rows}
    failures = []
    if len(rows) != 11 or observed_datasets != expected_datasets:
        failures.append(
            f"final_results.csv must contain exactly the 11 canonical datasets; got {sorted(observed_datasets)}"
        )
    for row in rows:
        if row["config"] not in CONFIG_HASHES:
            failures.append(f"unknown config in final_results.csv: {row['config']}")
        if row["seed_count"] != "5":
            failures.append(f"{row['dataset']} has seed_count={row['seed_count']}, expected 5")
        for name in (
            "cf_rmse_mean", "cf_rmse_std", "pehe_mean", "pehe_std",
            "po_rmse_mean", "po_rmse_std", "w1_mean", "w1_std",
            "w1_mu_mean", "w1_mu_std",
        ):
            value = float(row[name])
            if not math.isfinite(value) or value < 0:
                failures.append(f"invalid {name} for {row['dataset']}: {row[name]}")
    return failures


def main() -> None:
    failures = []
    failures.extend(verify_hashes(CORE_HASHES))
    failures.extend(verify_hashes(CONFIG_HASHES))
    failures.extend(verify_hashes(DATA_HASHES))
    failures.extend(verify_results())
    if failures:
        raise SystemExit("Release verification failed:\n- " + "\n- ".join(failures))
    print(
        "Release verified: "
        f"{len(CORE_HASHES)} frozen code files, "
        f"{len(CONFIG_HASHES)} final configs, "
        f"{len(DATA_HASHES)} processed archives, and 11 five-seed result rows."
    )


if __name__ == "__main__":
    main()
