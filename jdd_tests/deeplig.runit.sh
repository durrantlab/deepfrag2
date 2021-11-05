echo START: $(date)
python deeplig.py --csv ../data/moad/every.csv --data ../data/moad/BindingMoad2019/ --cache ./cache.json --gpus 1 --wandb_project 3aee432b3e7c672a3b2d2accf15b6b56a2770584 --num_sanity_val_steps 0
