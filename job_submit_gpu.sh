#!/bin/bash
# SBATCH -N 1                          # Number of nodes
#SBATCH -G 1                # Number of GPUs
#SBATCH --job-name=fel_model_gpu
#SBATCH --partition=ampere           # GPU partition
#SBATCH --account=ad:ard-online@ampere  # SLAC account
#SBATCH --cpus-per-task=8             # CPU cores for data loading (num_workers)
#SBATCH --mem=256G                    # Memory per node
#SBATCH -t 72:00:00                 # Time limit
#SBATCH --output=./slurm_outputs/output_%j.out
#SBATCH --error=./slurm_outputs/output_%j.log
# SBATCH --mail-type=ALL
# SBATCH --mail-user=zihanzhu@slac.stanford.edu

# Setup environment
export CXX=/usr/bin/g++
export CC=/usr/bin/gcc
export LD_LIBRARY_PATH=/sdf/data/ad/ard/u/zihanzhu/miniconda3/envs/ml_gpu/lib:$LD_LIBRARY_PATH
# Multi-threading for data loading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "========================================================================"
echo "SLURM JOB INFORMATION"
echo "========================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Submit dir:   $SLURM_SUBMIT_DIR"
echo "Node:         $HOSTNAME"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Memory:       $SLURM_MEM_PER_NODE MB"
echo "Start time:   $(date)"
echo

# ============================================================================
# GPU INFO
# ============================================================================
echo "========================================================================"
echo "GPU INFORMATION"
echo "========================================================================"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
echo
nvidia-smi
echo

python - <<'PYCODE'
import torch
print("Python OK, torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PYCODE

python train_fel_model.py \
    --epochs 620 \
    --batch_size 256 \
    --subsample_step 1 \
    --resume_from /sdf/scratch/users/z/zihanzhu/fel_tuning/checkpoints/19650505/epoch_600.pt

echo "End time:     $(date)"