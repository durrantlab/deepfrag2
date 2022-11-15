#!/bin/bash
#SBATCH --job-name=deepfrag
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
python MainDF2.py --data /ihome/jdurrant/crg93/prots_and_ligs/ --save_splits /ihome/jdurrant/crg93/output_ft_random/splits.json --default_root_dir /ihome/jdurrant/crg93/output_ft_random/ --aggregation_3x3_patches mean --aggregation_loss_vector mean --max_epochs 30 --save_every_epoch --mode warm_starting --model_for_warm_starting /ihome/jdurrant/crg93/output_deepfrag/model_mean_mean_train.pt
#python MainDF2.py --data /ihome/jdurrant/crg93/prots_and_ligs/ --save_splits /ihome/jdurrant/crg93/output_ft_buti04/splits.json --default_root_dir /ihome/jdurrant/crg93/output_ft_buti04/ --aggregation_3x3_patches mean --aggregation_loss_vector mean --max_epochs 30 --save_every_epoch --mode warm_starting --model_for_warm_starting /ihome/jdurrant/crg93/output_deepfrag/model_mean_mean_train.pt --butina_cluster_division --butina_cluster_cutoff 0.4