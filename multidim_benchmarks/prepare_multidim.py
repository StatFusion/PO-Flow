"""Build reproducible vector-outcome PO-Flow benchmark archives.

The tabular benchmarks stack potential-outcome surfaces evaluated on the same
covariates.  MorphoMNIST uses a semi-synthetic binary intensity intervention:
the observed handwriting supplies the unit-level shape/noise, while the two
potential images apply fixed low/high intensities to that same shape.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path: Path, *, metadata: dict, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    metadata = {**metadata, "archive": path.name, "sha256": _sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 2**20:.2f} MiB)")


def _split(n: int, treatment: np.ndarray, seed: int, test_fraction: float = 0.2):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for arm in (0, 1):
        indices = np.flatnonzero(treatment == arm)
        rng.shuffle(indices)
        cut = max(1, int(round(len(indices) * test_fraction)))
        test.extend(indices[:cut])
        train.extend(indices[cut:])
    return np.sort(np.asarray(train)), np.sort(np.asarray(test))


def prepare_ihdp(dimensions: tuple[int, ...], seed: int) -> None:
    raw = ROOT / "raw" / "ihdp"
    files = [raw / f"ihdp_npci_{index}.csv" for index in range(1, 11)]
    frames = [pd.read_csv(path, header=None) for path in files]
    reference_a = frames[0].iloc[:, 0].to_numpy(dtype=np.int64)
    reference_x = frames[0].iloc[:, 5:].to_numpy(dtype=np.float32)
    for path, frame in zip(files[1:], frames[1:]):
        if not np.array_equal(frame.iloc[:, 0].to_numpy(), reference_a):
            raise ValueError(f"treatment mismatch in {path}")
        if not np.allclose(frame.iloc[:, 5:].to_numpy(), reference_x):
            raise ValueError(f"covariate mismatch in {path}")

    def recover(frame: pd.DataFrame):
        a = frame.iloc[:, 0].to_numpy(dtype=np.int64)
        factual = frame.iloc[:, 1].to_numpy(dtype=np.float32)
        counterfactual = frame.iloc[:, 2].to_numpy(dtype=np.float32)
        y0 = np.where(a == 0, factual, counterfactual)
        y1 = np.where(a == 1, factual, counterfactual)
        return (
            y0,
            y1,
            frame.iloc[:, 3].to_numpy(dtype=np.float32),
            frame.iloc[:, 4].to_numpy(dtype=np.float32),
        )

    surfaces = [recover(frame) for frame in frames]
    train, test = _split(len(reference_a), reference_a, seed)
    for dimension in dimensions:
        selected = surfaces[:dimension]
        _save(
            ROOT / "processed" / f"ihdp_vector_d{dimension}.npz",
            metadata={
                "name": "IHDP-Vector",
                "construction": "Stack the first d CEVAE/NPCI IHDP replications; all share X and A.",
                "dimension": dimension,
                "source": "https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP/csv",
                "seed": seed,
                "n": len(reference_a),
                "x_dim": reference_x.shape[1],
            },
            A=reference_a,
            X=reference_x,
            Y0=np.column_stack([surface[0] for surface in selected]),
            Y1=np.column_stack([surface[1] for surface in selected]),
            MU0=np.column_stack([surface[2] for surface in selected]),
            MU1=np.column_stack([surface[3] for surface in selected]),
            train_idx=train,
            test_idx=test,
        )


def prepare_acic(dimensions: tuple[int, ...], seed: int) -> None:
    raw = ROOT / "raw" / "acic2016"
    x_frame = pd.read_csv(raw / "x.csv")
    x = pd.get_dummies(x_frame, drop_first=False, dtype=float).to_numpy(dtype=np.float32)
    frames = [pd.read_csv(raw / f"zymu_{index}.csv") for index in range(1, 11)]
    if any(len(frame) != len(x) for frame in frames):
        raise ValueError("ACIC covariates and potential-outcome surfaces are not row aligned")
    # ACIC repetitions use different assignment draws.  A vector potential
    # outcome remains well-defined by fixing the first draw and stacking only
    # the disclosed Y(0),Y(1) surfaces from the other repetitions.
    treatment = frames[0]["z"].to_numpy(dtype=np.int64)
    train, test = _split(len(treatment), treatment, seed)
    for dimension in dimensions:
        selected = frames[:dimension]
        _save(
            ROOT / "processed" / f"acic2016_vector_d{dimension}.npz",
            metadata={
                "name": "ACIC2016-Vector",
                "construction": "Fix replication-1 A; stack Y(0),Y(1) surfaces from the first d replications.",
                "dimension": dimension,
                "source": "https://github.com/BiomedSciAI/causallib/tree/master/causallib/datasets/data/acic_challenge_2016",
                "seed": seed,
                "n": len(treatment),
                "x_dim": x.shape[1],
            },
            A=treatment,
            X=x,
            Y0=np.column_stack([frame["y0"].to_numpy(np.float32) for frame in selected]),
            Y1=np.column_stack([frame["y1"].to_numpy(np.float32) for frame in selected]),
            MU0=np.column_stack([frame["mu0"].to_numpy(np.float32) for frame in selected]),
            MU1=np.column_stack([frame["mu1"].to_numpy(np.float32) for frame in selected]),
            train_idx=train,
            test_idx=test,
        )


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, size = struct.unpack(">II", handle.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", handle.read(8))
            return np.frombuffer(handle.read(), dtype=np.uint8).reshape(size, rows, cols)
        if magic == 2049:
            return np.frombuffer(handle.read(), dtype=np.uint8).reshape(size)
        raise ValueError(f"unsupported IDX magic {magic} in {path}")


def prepare_morphomnist(seed: int, limit: int | None) -> None:
    raw = ROOT / "raw" / "morphomnist"
    image_parts, label_parts, metric_parts = [], [], []
    for prefix in ("train", "t10k"):
        image_parts.append(_read_idx(raw / f"{prefix}-images-idx3-ubyte.gz"))
        label_parts.append(_read_idx(raw / f"{prefix}-labels-idx1-ubyte.gz"))
        metric_parts.append(pd.read_csv(raw / f"{prefix}-morpho.csv"))
    images = np.concatenate(image_parts)
    labels = np.concatenate(label_parts)
    metrics = pd.concat(metric_parts, ignore_index=True)
    if limit is not None:
        images, labels, metrics = images[:limit], labels[:limit], metrics.iloc[:limit]

    observed_intensity = metrics["intensity"].to_numpy(np.float32)
    thickness = metrics["thickness"].to_numpy(np.float32)
    shape = np.clip(images.astype(np.float32) / np.maximum(observed_intensity[:, None, None], 1.0), 0.0, 1.0)
    low_intensity, high_intensity = 110.0, 220.0
    y0 = np.rint(shape * low_intensity).clip(0, 255).astype(np.uint8).reshape(len(images), -1)
    y1 = np.rint(shape * high_intensity).clip(0, 255).astype(np.uint8).reshape(len(images), -1)

    digit_one_hot = np.eye(10, dtype=np.float32)[labels]
    thickness_z = ((thickness - thickness.mean()) / thickness.std()).astype(np.float32)
    x = np.column_stack([digit_one_hot, thickness_z]).astype(np.float32)
    logits = 0.55 * thickness_z + 0.12 * (labels.astype(np.float32) - 4.5)
    propensity = 1.0 / (1.0 + np.exp(-logits))
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, propensity).astype(np.int64)
    train, test = _split(len(treatment), treatment, seed)

    _save(
        ROOT / "processed" / "morphomnist_intensity_d784.npz",
        metadata={
            "name": "MorphoMNIST-Intensity",
            "construction": "Shared handwriting shape with binary do(intensity=110 or 220) potential images.",
            "dimension": 784,
            "source": "https://github.com/gulnazaki/counterfactual-benchmark/tree/main/counterfactual_benchmark/datasets/morphomnist",
            "seed": seed,
            "n": len(treatment),
            "x_dim": x.shape[1],
            "storage": "Y arrays are uint8 and are scaled to [-1,1] by the training loader.",
        },
        A=treatment,
        X=x,
        Y0=y0,
        Y1=y1,
        MU0=y0,
        MU1=y1,
        propensity=propensity.astype(np.float32),
        train_idx=train,
        test_idx=test,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="2,5,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--morpho-limit", type=int, default=None)
    parser.add_argument("--skip-morphomnist", action="store_true")
    args = parser.parse_args()
    dimensions = tuple(int(value) for value in args.dimensions.split(","))
    if not dimensions or min(dimensions) < 1 or max(dimensions) > 10:
        raise ValueError("dimensions must lie between 1 and 10")
    prepare_ihdp(dimensions, args.seed)
    prepare_acic(dimensions, args.seed)
    if not args.skip_morphomnist:
        prepare_morphomnist(args.seed, args.morpho_limit)


if __name__ == "__main__":
    main()
