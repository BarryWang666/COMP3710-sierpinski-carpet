#!/bin/bash
#SBATCH --job-name=sierpinski_depth_progression
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=outputs/sierpinski_depth_progression_%j.out
#SBATCH --error=outputs/sierpinski_depth_progression_%j.err

cd "$SLURM_SUBMIT_DIR"

echo "hostname: $(hostname)"
echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm submit directory: $SLURM_SUBMIT_DIR"
nvidia-smi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

export MPLBACKEND=Agg

python src/depth_progression.py
