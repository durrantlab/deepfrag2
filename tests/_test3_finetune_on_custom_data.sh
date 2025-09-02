echo "Finetune on custom data"

mkdir -p 3.finetune_custom.output

python -u ../MainDF2.py \
    --mode warm_starting \
    --max_epochs 5 \
    --save_params ./3.finetune_custom.output/params.saved.json \
    --save_splits ./3.finetune_custom.output/splits.saved.json \
    --model_for_warm_starting ./1.train_on_moad.output/model_train_last.pt \
    --csv ./data_to_finetune/pdb_sdf_file_pairs.csv \
    --data_dir ./data_to_finetune/ \
    --cache ./3.finetune_custom.output/every_csv.cache.json \
    --default_root_dir $(pwd)/3.finetune_custom.output/  `# The output directory` \
    --json_params common_params.json.inp \
    --split_method butina \
    --butina_cluster_cutoff 0.4 \
    --fraction_val 0.0  `# No validation set` \
    --cache_pdbs_to_disk \
    | tee 3.OUT-python_out.txt
