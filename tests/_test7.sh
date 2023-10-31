. _init.sh

echo "Train on a small subset of the Binding MOAD, only larger fragments that are aromatic"

mkdir -p 7.train_on_moad_large_aromatic_frags.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode train \
    --max_epochs 3 \
    --save_params ./7.train_on_moad_large_aromatic_frags.output/params.saved.json \
    --save_splits ./7.train_on_moad_large_aromatic_frags.output/splits.saved.json \
    --every_csv $MOAD_DIR/${EVERY_CSV_BSNM} \
    --cache ./every_csv.cache.json \
    --data_dir $MOAD_DIR/ \
    --default_root_dir $(pwd)/7.train_on_moad_large_aromatic_frags.output/  `# The output directory` \
    --max_pdbs_train 100  `# Train on a small subset of the Binding MOAD` \
    --max_pdbs_val 100  `# Validate on a small subset of the Binding MOAD` \
    --json_params common_params.json.inp \
    --min_frag_num_heavy_atoms 4 \
    --mol_props aromatic \
    | tee 7.OUT-python_out.txt
