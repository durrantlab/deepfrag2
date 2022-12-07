#!/bin/bash
#SBATCH --job-name=deepfrag_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --time=144:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

module purge
module load python/ondemand-jupyter-python3.8

source activate deepfrag2cesar

cd /ihome/jdurrant/crg93/deepfrag2/

python MainDF2.py --csv /bgfs/jdurrant/durrantj/deepfrags/deepfrag2_post_cesar/data/moad/every.csv --data /bgfs/jdurrant/durrantj/deepfrags/deepfrag2_post_cesar/data/moad/BindingMOAD_2020/ --save_splits /ihome/jdurrant/crg93/output_deepfrag/splits.json --default_root_dir /ihome/jdurrant/crg93/output_deepfrag/ --aggregation_3x3_patches mean --aggregation_loss_vector mean --max_epochs 30
