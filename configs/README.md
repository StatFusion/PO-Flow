# Canonical final configurations

This directory is the single configuration entry point for the final
overlap-aware PO-Flow experiments. It contains exactly 11 PO-Flow YAML files:
four scalar outcomes under `scalar/` and seven vector/image outcomes under
`vector/`. The two files under `baselines/` freeze all matched baseline
settings. Search grids, pilot runs, and rejected candidates are intentionally
excluded.

Every PO-Flow YAML begins with the exact deanonymized cluster setup, the
five-seed runner command, and a direct single-seed call. The reported seeds are
42, 43, 44, 45, and 46.

## Provenance

The scalar configurations are the safety-selected final IPW configurations:

| Configuration | Selection | q | alpha | Epochs / batch |
|---|---|---:|---:|---:|
| `scalar/ACIC2016.yaml` | retained incumbent | .99 | .25 | 310 / 1000 |
| `scalar/ACIC2018.yaml` | `c04_q95_a075` | .95 | .75 | 150 / 200 |
| `scalar/IHDP.yaml` | `c01_q95_a010` | .95 | .10 | 90 / 150 |
| `scalar/IBM.yaml` | retained incumbent | .95 | 1.00 | 270 / 200 |

The vector/image configurations are the final optimization-v3 release. The
IHDP configurations retain architecture-v2 after the prespecified safety gate;
ACIC16 uses the final joint `(q, alpha)=(.95,.75)` setting; MorphoMNIST uses the
final covariate-standardized, half-batch setting. No v3 candidate increases
network capacity.

The exact aggregate metrics, original candidate labels, and mean run times are
in `reproducibility/final_results.csv`. `python verify_release.py` checks every
config against the SHA-256 fingerprint recorded at the time of the formal
runs.

The root-level `ACIC2016.yaml`, `ACIC2018.yaml`, `IHDP.yaml`, and `IBM.yaml`
remain the original unweighted conference configurations. The former
root-level `*_IPW.yaml` files are redundant parsed copies and are ignored.
Likewise, `multidim_benchmarks/configs/` contains superseded pre-v2 copies and
is ignored.

## Frozen PO-Flow network sizes

Counts exclude the separately fitted propensity model and do not double-count
the SWA evaluation copy.

| Configuration | dX | dY | Network | Trainable parameters |
|---|---:|---:|---|---:|
| `scalar/ACIC2016.yaml` | 82 | 1 | scalar residual, c=83, h=128, L=2 | 136,008 |
| `scalar/ACIC2018.yaml` | 177 | 1 | scalar residual, c=178, h=128, L=2 | 595,618 |
| `scalar/IHDP.yaml` | 25 | 1 | scalar residual, c=26, h=128, L=2 | 16,194 |
| `scalar/IBM.yaml` | 177 | 1 | scalar residual, c=178, h=128, L=2 | 595,618 |
| `vector/IHDP_Vector_d2.yaml` | 25 | 2 | time residual, w=192, L=4 | 1,014,340 |
| `vector/IHDP_Vector_d5.yaml` | 25 | 5 | time residual, w=192, L=3 | 793,546 |
| `vector/IHDP_Vector_d10.yaml` | 25 | 10 | time residual, w=192, L=3 | 796,436 |
| `vector/ACIC2016_Vector_d2.yaml` | 82 | 2 | time residual, w=192, L=3 | 802,756 |
| `vector/ACIC2016_Vector_d5.yaml` | 82 | 5 | time residual, w=192, L=3 | 804,490 |
| `vector/ACIC2016_Vector_d10.yaml` | 82 | 10 | time residual, w=192, L=4 | 1,029,908 |
| `vector/MorphoMNIST_Intensity_d784.yaml` | 11 | 784 | time residual, w=512, L=3 | 6,747,680 |

All time-residual configurations use a 32-dimensional time embedding,
expansion factor 2, and separate treatment heads. The ACIC16 auxiliary mean
readout reuses the same network at an out-of-flow task token and adds no
parameters.

## Complete final call

After the setup block at the top of any YAML:

```bash
python run_tuned_ipw.py run \
  --datasets ACIC2016,ACIC2018,IHDP,IBM,IHDP_Vector_d2,IHDP_Vector_d5,IHDP_Vector_d10,ACIC2016_Vector_d2,ACIC2016_Vector_d5,ACIC2016_Vector_d10,MorphoMNIST_Intensity_d784 \
  --seeds 42,43,44,45,46 --jobs 1
python run_tuned_ipw.py summarize
```

Increase `--jobs` only when the allocated GPU resources can support concurrent
processes.
