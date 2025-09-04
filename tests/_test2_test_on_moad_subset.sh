echo "Test on a small subset of the Binding MOAD (--max_pdbs_test 100)"

mkdir -p 2.test_moad_trained.output

python -u ../MainDF2.py \
    --mode test_on_moad \
    --load_splits ./1.train_on_moad.output/splits.saved.json \
    --load_checkpoint ./1.train_on_moad.output/last.ckpt \
    --csv moad/every.csv \
    --data_dir moad/ \
    --cache ./1.train_on_moad.output/every_csv.cache.json \
    --default_root_dir $(pwd)/2.test_moad_trained.output/  `# The output directory` \
    --inference_label_sets test \
    --rotations 2 \
    --max_pdbs_test 100 \
    --json_params common_params.json.inp \
    --cache_pdbs_to_disk \
    | tee 2.OUT-python_out.txt
