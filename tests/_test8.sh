. _init.sh

echo "Test on a small subset of the Binding MOAD (--max_pdbs_test 100)"

mkdir -p 8.test_big_aromatic_trained.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode test \
    --load_splits ./7.train_on_moad_large_aromatic_frags.output/splits.saved.json \
    --load_checkpoint ./7.train_on_moad_large_aromatic_frags.output/last.ckpt \
    --every_csv $MOAD_DIR/${EVERY_CSV_BSNM} \
    --cache ./every_csv.cache.json \
    --data_dir $MOAD_DIR/ \
    --default_root_dir $(pwd)/8.test_big_aromatic_trained.output/  `# The output directory` \
    --inference_label_sets test \
    --rotations 2 \
    --max_pdbs_test 100 \
    --json_params common_params.json.inp \
    | tee 8.OUT-python_out.txt
