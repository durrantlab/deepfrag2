#!/bin/bash
#SBATCH --job-name=deepfrag_assess_everymodel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --time=144:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --array=0-29

module purge
module load python/ondemand-jupyter-python3.8

. paths.sh

source activate ${CONDA_ENV_NAME}

cd ${OUTPUT_DIR}/

if [ $SLURM_ARRAY_TASK_ID -lt 10 ]; then
  file=`ls loss-epoch=0${SLURM_ARRAY_TASK_ID}*.ckpt`
else
  file=`ls loss-epoch=${SLURM_ARRAY_TASK_ID}*.ckpt`
fi

file=$(realpath $file)
echo ${file}

cd ${DEEPFRAG_ROOT}/

operator="mean"

python MainDF2.py --data ${DIR_WITH_MODELS}/ --load_splits ${OUTPUT_DIR}/splits.json --default_root_dir ${OUTPUT_DIR} --mode test --inference_rotations 8 --aggregation_rotations ${operator} --load_checkpoint ${file}
