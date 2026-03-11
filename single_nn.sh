#!/bin/bash
#SBATCH --partition=milano
#SBATCH --job-name=NN_evaluate
#SBATCH --account=ad:beamphysics
#SBATCH -t 144:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=./slurm_outputs/evaluate_nn_%j.out
#SBATCH --error=./slurm_outputs/evaluate_nn_%j.err

# Usage: sbatch run_archiver_by_month.slurm START_DATE END_DATE
# Assumes you have already activated the correct conda/env so that `python` has all deps.

source /sdf/data/ad/ard/u/zihanzhu/miniconda3/bin/activate ml_gpu
python model_evaluation.py