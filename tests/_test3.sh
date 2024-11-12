. _init.sh

echo "Finetune on custom data"

mkdir -p 3.finetune_moad.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode warm_starting \
    --max_epochs 5 \
    --save_params ./3.finetune_moad.output/params.saved.json \
    --save_splits ./3.finetune_moad.output/splits.saved.json \
    --model_for_warm_starting ./1.train_on_moad.output/model_train.pt \
    --data_dir ./data_to_finetune/  `# Protein/ligands named like 1XDN_prot_123.pdb, 1XDN_lig_123.sdf` \
    --cache ./3.finetune_moad.output/every_csv.cache.json \
    --default_root_dir $(pwd)/3.finetune_moad.output/  `# The output directory` \
    --json_params common_params.json.inp \
    --split_method butina \
    --butina_cluster_cutoff 0.4 \
    --fraction_val 0.0  `# No validation set` \
    --cache_pdbs_to_disk \
    | tee 3.OUT-python_out.txt
