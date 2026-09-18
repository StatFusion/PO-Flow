#!/usr/bin/env bash
set -eo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$project_dir"

mkdir -p results/multidim_baselines/logs NN_params/multidim_baselines

configs=(
  configs/vector/IHDP_Vector_d2.yaml
  configs/vector/ACIC2016_Vector_d2.yaml
  configs/vector/IHDP_Vector_d5.yaml
  configs/vector/ACIC2016_Vector_d5.yaml
  configs/vector/IHDP_Vector_d10.yaml
  configs/vector/ACIC2016_Vector_d10.yaml
  configs/vector/MorphoMNIST_Intensity_d784.yaml
)
methods=(cfr ganite cevae)
seeds=(42 43 44 45 46)

for config in "${configs[@]}"; do
  dataset=$(basename "$config" .yaml)
  for method in "${methods[@]}"; do
    method_upper=$(printf '%s' "$method" | tr '[:lower:]' '[:upper:]')
    for seed in "${seeds[@]}"; do
      run_name="${dataset}_${method_upper}_seed${seed}"
      metric="results/multidim_baselines/${run_name}_metrics.csv"
      log="results/multidim_baselines/logs/${run_name}.log"
      if [[ -s "$metric" ]]; then
        printf 'SKIP completed %s\n' "$run_name"
        continue
      fi
      printf 'START %s %s\n' "$(date --iso-8601=seconds)" "$run_name"
      python multidim_baselines.py \
        --config "$config" \
        --baseline-config configs/baselines/Multidim_Baselines.yaml \
        --method "$method" --device cuda --seed "$seed" --run-name "$run_name" \
        2>&1 | tee "$log"
      test -s "$metric"
      printf 'DONE %s %s\n' "$(date --iso-8601=seconds)" "$run_name"
    done
  done
done

python summarize_multidim_baselines.py
