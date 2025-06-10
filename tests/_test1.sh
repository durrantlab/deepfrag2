. _init.sh

echo "Train on a small subset of the Binding MOAD (for testing; --max_pdbs_train 100, --max_pdbs_val 100)"

mkdir -p 1.train_on_moad.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode train_on_moad \
    --max_epochs 3 \
    --save_params ./1.train_on_moad.output/params.saved.json \
    --save_splits ./1.train_on_moad.output/splits.saved.json \
    --csv $MOAD_DIR/${EVERY_CSV_BSNM} \
    --data_dir $MOAD_DIR/ \
    --cache ./1.train_on_moad.output/every_csv.cache.json \
    --default_root_dir $(pwd)/1.train_on_moad.output/  `# The output directory` \
    --max_pdbs_train 100  `# Train on a small subset of the Binding MOAD` \
    --max_pdbs_val 100  `# Validate on a small subset of the Binding MOAD` \
    --json_params common_params.json.inp \
    --cache_pdbs_to_disk \
    | tee 1.OUT-python_out.txt
