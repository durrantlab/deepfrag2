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
params="mean owa1 owa2 owa3 owa_exp_smooth1_01 owa_exp_smooth1_02 owa_exp_smooth1_03 owa_exp_smooth1_04 owa_exp_smooth1_05 owa_exp_smooth1_06 owa_exp_smooth1_07 owa_exp_smooth1_08 owa_exp_smooth1_09 owa_exp_smooth2_01 owa_exp_smooth2_02 owa_exp_smooth2_03 owa_exp_smooth2_04 owa_exp_smooth2_05 owa_exp_smooth2_06 owa_exp_smooth2_07 owa_exp_smooth2_08 owa_exp_smooth2_09 choquet_integral_cf choquet_integral_symmetric sugeno_fuzzy_integral"
for param in $params;
do
  python MainDF2.py --csv /bgfs/jdurrant/durrantj/deepfrags/deepfrag2_post_cesar/data/moad/every.csv --data /bgfs/jdurrant/durrantj/deepfrags/deepfrag2_post_cesar/data/moad/BindingMOAD_2020/ --load_splits /ihome/jdurrant/crg93/output_deepfrag/splits.json --default_root_dir /ihome/jdurrant/crg93/output_deepfrag --mode test --inference_rotations 8 --aggregation_rotations ${param} --load_newest_checkpoint True
done
