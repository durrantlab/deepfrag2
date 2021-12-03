export NCCL_ASYNC_ERROR_HANDLING=1
#wandb login
echo START: $(date)
# 
CUDA_LAUNCH_BLOCKING=1 NCCL_ASYNC_ERROR_HANDLING=1 python deeplig.py --csv ../data/moad/every.csv --data ../data/moad/BindingMoad2019/ --cache ./cache.json --gpus 1 --num_sanity_val_steps 0 --num_workers 0
#16

# --wandb_project 3aee432b3e7c672a3b2d2accf15b6b56a2770584 --num_workers 1
