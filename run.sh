#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --partition=gpu-v100-16g
# SBATCH --partition=gpu-h100-80g-short
# SBATCH --partition=gpu-h200-141g-short
# SBATCH --partition=gpu-debug
# SBATCH --partition=gpu-a100-80g 
# SBATCH --partition=gpu-amd
# SBATCH --gres=gpu-vram:64g
# SBATCH -o /home/sethih1/masque_new/ters_gen/log_file/slurm_%j.out


# Load Environments

module load scicomp-python-env/2024-01
source activate /scratch/work/sethih1/Crow/chemcrow-env/

python --version

python run.py