echo START: $(date)

python deeplig.py \
    --every_csv ../../data/moad/every.csv \
    --data_dir ../../data/moad/BindingMoad2019/ \
    --cache ../cache.json \
    --gpus 1 \
    --wandb_project 3aee432b3e7c672a3b2d2accf15b6b56a2770584 \
    --num_dataloader_workers 16 \
    --max_voxels_in_memory 320

    # --num_sanity_val_steps 0 \
    # --cpu
