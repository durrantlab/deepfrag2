echo "Finetune on custom data (with pt download)"

mkdir -p 13.finetune_custom_download.output

python -u ../MainDF2.py \
    --mode warm_starting \
    --max_epochs 5 \
    --save_params ./13.finetune_custom_download.output/params.saved.json \
    --save_splits ./13.finetune_custom_download.output/splits.saved.json \
    --model_for_warm_starting gte_4_last_for_finetuning \
    --csv ./data_to_finetune/pdb_sdf_file_pairs.csv \
    --data_dir ./data_to_finetune/ \
    --cache ./13.finetune_custom_download.output/every_csv.cache.json \
    --default_root_dir $(pwd)/13.finetune_custom_download.output/  `# The output directory` \
    --json_params common_params.json.inp \
    --split_method butina \
    --butina_cluster_cutoff 0.4 \
    --fraction_val 0.0  `# No validation set` \
    --cache_pdbs_to_disk \
    | tee 13.OUT-python_out.txt
