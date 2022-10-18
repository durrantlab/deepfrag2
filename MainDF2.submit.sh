#!/bin/bash
#SBATCH --job-name=deepfrag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --time=78:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

. /ihome/jdurrant/durrantj/.bashrc

module purge
module load cuda/11.1.1

export PATH=/ihome/jdurrant/durrantj/miniconda3/envs/deepfrag2/bin:/ihome/jdurrant/durrantj/miniconda3/condabin:${PATH}
conda activate deepfrag2
export LD_LIBRARY_PATH=/ihome/jdurrant/durrantj/miniconda3/envs/deepfrag2/lib:$LD_LIBRARY_PATH

python MainDF2.py

