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

module purge
module load cuda/11.1.1
module load pytorch_gpu/1.11.0

source activate deepfrag2cesar

cd /ihome/jdurrant/crg93/deepfrag2/
python MainDF2.py --csv /ihome/jdurrant/crg93/moad/every.csv --data /ihome/jdurrant/crg93/moad/BindingMOAD_2020 --save_splits /ihome/jdurrant/crg93/output/splits.json --default_root_dir /ihome/jdurrant/crg93/output --aggregation_3x3_patches mean --aggregation_loss_vector mean --max_epochs 30
