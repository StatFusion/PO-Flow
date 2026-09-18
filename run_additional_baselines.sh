#!/usr/bin/env bash
set -eo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$project_dir"

mkdir -p results/additional_baselines/logs NN_params/additional_baselines

configs=(
  configs/scalar/ACIC2016.yaml
  configs/scalar/ACIC2018.yaml
  configs/scalar/IHDP.yaml
  configs/scalar/IBM.yaml
  configs/vector/IHDP_Vector_d2.yaml
  configs/vector/ACIC2016_Vector_d2.yaml
  configs/vector/IHDP_Vector_d5.yaml
  configs/vector/ACIC2016_Vector_d5.yaml
  configs/vector/IHDP_Vector_d10.yaml
  configs/vector/ACIC2016_Vector_d10.yaml
  configs/vector/MorphoMNIST_Intensity_d784.yaml
)
methods=(tarnet dragonnet)
seeds=(42 43 44 45 46)

for config in "${configs[@]}"; do
  dataset=$(basename "$config" .yaml)
  for method in "${methods[@]}"; do
    if [[ "$method" == "tarnet" ]]; then
      method_label=TARNet
    else
      method_label=DragonNet
    fi
    for seed in "${seeds[@]}"; do
      run_name="${dataset}_${method_label}_seed${seed}"
      metric="results/additional_baselines/${run_name}_metrics.csv"
      metadata="results/additional_baselines/${run_name}_metadata.json"
      checkpoint="NN_params/additional_baselines/${run_name}.pth"
      log="results/additional_baselines/logs/${run_name}.log"
      if [[ -s "$metric" && -s "$metadata" && -s "$checkpoint" ]]; then
        printf 'SKIP completed %s\n' "$run_name"
        continue
      fi
      printf 'START %s %s\n' "$(date --iso-8601=seconds)" "$run_name"
      python additional_baselines.py \
        --config "$config" \
        --baseline-config configs/baselines/Neural_Point_Baselines.yaml \
        --method "$method" --device cuda --seed "$seed" --run-name "$run_name" \
        2>&1 | tee "$log"
      test -s "$metric"
      printf 'DONE %s %s\n' "$(date --iso-8601=seconds)" "$run_name"
    done
  done
done

python summarize_additional_baselines.py
python audit_additional_baselines.py
