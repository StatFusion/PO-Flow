# Vector and image benchmark archives

The `processed/` directory contains the exact deterministic archives consumed
by all seven final files in `configs/vector/`. They are versioned so a clean
checkout can launch the final experiments without reconstructing third-party
source datasets.

| Archive family | Construction |
|---|---|
| IHDP-Vector | Stack the first d in {2,5,10} aligned CEVAE/NPCI response-surface replications. |
| ACIC16-Vector | Fix the replication-1 treatment assignment and stack the first d disclosed potential-outcome surfaces. |
| MorphoMNIST-Intensity | Preserve each handwriting shape and construct paired `do(intensity=110)` and `do(intensity=220)` images. |

All treatment-stratified train/test splits use seed 42. The adjacent JSON files
record the source, construction, dimensions, and SHA-256 digest of each NPZ.
`python verify_release.py` validates those digests.

`prepare_multidim.py` is the deterministic builder used to create the
archives. To regenerate them, place the upstream files in the paths expected
under `raw/` and run:

```bash
python multidim_benchmarks/prepare_multidim.py --dimensions 2,5,10 --seed 42
```

The raw third-party downloads are intentionally not versioned. Their sources
are recorded in `prepare_multidim.py` and in the processed JSON metadata.
