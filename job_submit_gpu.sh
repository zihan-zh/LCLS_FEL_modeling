#!/bin/bash
# SBATCH -N 1                          # Number of nodes
#SBATCH -G 1                # Number of GPUs
#SBATCH --job-name=fel_model
#SBATCH --partition=ampere           # GPU partition
#SBATCH --account=ad:default@ampere  # SLAC account
#SBATCH --mem=256G                    # Memory per node
#SBATCH -t 72:00:00                 # Time limit
#SBATCH --output=./slurm_outputs/output_%j.out
#SBATCH --error=./slurm_outputs/output_%j.err
# SBATCH --mail-type=ALL
# SBATCH --mail-user=zihanzhu@slac.stanford.edu

# Setup environment
export CXX=/usr/bin/g++
export CC=/usr/bin/gcc
export LD_LIBRARY_PATH=/sdf/data/ad/ard/u/zihanzhu/miniconda3/envs/ml_gpu/lib:$LD_LIBRARY_PATH
echo "Job ID:       $SLURM_JOB_ID"
echo "Submit dir:   $SLURM_SUBMIT_DIR"
echo "Running on:   $HOSTNAME"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPUs visible: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo
nvidia-smi
# Quick check that Python & Torch see the GPU
python - <<'PYCODE'
import torch
print("Python OK, torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PYCODE

python train_fel_model.py \
    --epochs 300 \
    --batch_size 256 \
    --subsample_step 1