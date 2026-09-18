#!/bin/bash
#SBATCH --job-name=poflow_extra_baselines
#SBATCH --account=gts-yxie77-paid
#SBATCH --partition=gpu-v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-gpu=40G
#SBATCH --time=01:00:00
#SBATCH --qos=inferno
#SBATCH --output=/storage/home/hcoda1/3/dwu381/r-yxie77-0/PO_Flow/additional_baselines_slurm_%j.log
#SBATCH --error=/storage/home/hcoda1/3/dwu381/r-yxie77-0/PO_Flow/additional_baselines_slurm_%j.log

set -eo pipefail

module load anaconda3/2022.05
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /storage/home/hcoda1/3/dwu381/r-yxie77-0/dongze
cd /storage/home/hcoda1/3/dwu381/r-yxie77-0/PO_Flow

export PYTHONPYCACHEPREFIX=/tmp/poflow_additional_pycache

python -m pytest -q -p no:cacheprovider
bash run_additional_baselines.sh
python summarize_additional_baselines.py
