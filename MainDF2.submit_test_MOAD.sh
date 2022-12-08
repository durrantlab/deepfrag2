#!/bin/bash
#SBATCH --job-name=deepfrag_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --time=144:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --array=0-24

module purge
module load python/ondemand-jupyter-python3.8

. paths.sh

source activate ${CONDA_ENV_NAME}

cd ${DEEPFRAG_ROOT}/

params=("mean" "owa1" "owa2" "owa3" "owa_exp_smooth1_01" "owa_exp_smooth1_02" "owa_exp_smooth1_03" "owa_exp_smooth1_04" "owa_exp_smooth1_05" "owa_exp_smooth1_06" "owa_exp_smooth1_07" "owa_exp_smooth1_08" "owa_exp_smooth1_09" "owa_exp_smooth2_01" "owa_exp_smooth2_02" "owa_exp_smooth2_03" "owa_exp_smooth2_04" "owa_exp_smooth2_05" "owa_exp_smooth2_06" "owa_exp_smooth2_07" "owa_exp_smooth2_08" "owa_exp_smooth2_09" "choquet_integral_cf" "choquet_integral_symmetric" "sugeno_fuzzy_integral")

python MainDF2.py --csv ${EVERY_CSV} --data ${BINDINGMOAD_DIR}/ --load_splits ${OUTPUT_DIR}/splits.json --default_root_dir ${OUTPUT_DIR}/ --mode test --inference_rotations 8 --aggregation_rotations ${params[${SLURM_ARRAY_TASK_ID}]} --load_newest_checkpoint True
