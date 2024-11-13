. _init.sh

echo "Test on a small subset of the Binding MOAD (--max_pdbs_test 100)"

mkdir -p 2.test_moad_trained.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode test \
    --load_splits ./1.train_on_moad.output/splits.saved.json \
    --load_checkpoint ./1.train_on_moad.output/last.ckpt \
    --every_csv $MOAD_DIR/${EVERY_CSV_BSNM} \
    --data_dir $MOAD_DIR/ \
    --cache ./1.train_on_moad.output/every_csv.cache.json \
    --default_root_dir $(pwd)/2.test_moad_trained.output/  `# The output directory` \
    --inference_label_sets test \
    --rotations 2 \
    --max_pdbs_test 100 \
    --json_params common_params.json.inp \
    --cache_pdbs_to_disk \
    | tee 2.OUT-python_out.txt
