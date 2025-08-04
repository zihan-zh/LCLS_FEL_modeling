#!/bin/bash
#SBATCH --partition=ampere #milano         # Partition name (adjust based on your resources)
#SBATCH --job-name=fel_model
#SBATCH --account=ad:ard-online@ampere  # @ad:beamphysics    # Account name ad:beamphysics ad:ard-online@ampere  
# SBATCH --mail-user=zihanzhu@slac.stanford.edu
# SBATCH --mail-type=ALL            # Notifications for job start, end, and failure
#SBATCH -t 144:00:00                # Wall clock time limit
#SBATCH --output=./slurm_outputs/%j.out
#SBATCH --error=./slurm_outputs/%j.err
#SBATCH --mem=256G
#SBATCH --ntasks=1
# OpenMP settings
# export OMP_NUM_THREADS=4           # Number of threads
# export OMP_PLACES=threads
# export OMP_PROC_BIND=close
# Initialize Conda
source /sdf/group/ad/beamphysics/software/elegant/milano_openmpi/setup_elegant


echo -e "Bsub job ID: $LSB_JOBID"
echo -e "Working dir: $LS_SUBCWD"
# A little useful information for the log file...
echo -e "Master process running on: $HOSTNAME"
echo -e "Directory is:  $PWD"


python -c "print('Python environment is set up and working!')"

python train_fel_model.py --epochs 300 --batch_size 256 #--model_path ./autoencoder_20250220_1950.pth
# srun -n 1 -c 1 --cpu_bind=cores --gpu-bind=single:1 python train_test.py --dataset shapenet --epochs 30 \
# --batch_size 1 --output_path /sdf/data/ad/ard/u/nayaknil/Pointcloud_modeling/D-PCC/output_test_graph \
# --latent_xyzs_conv_mode edge_conv --sub_point_conv_mode edge_conv --k 4 --max_upsample_num 4 4 4