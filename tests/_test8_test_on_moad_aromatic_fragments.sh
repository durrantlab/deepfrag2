echo "Test on a small subset of the Binding MOAD (--max_pdbs_test 100)"

mkdir -p 8.test_big_aromatic_trained.output

python -u ../MainDF2.py \
    --mode test_on_moad \
    --load_splits ./7.train_on_moad_large_aromatic_frags.output/splits.saved.json \
    --load_checkpoint ./7.train_on_moad_large_aromatic_frags.output/last.ckpt \
    --csv moad/every.csv \
    --cache ./every_csv.cache.json \
    --data_dir moad/ \
    --default_root_dir $(pwd)/8.test_big_aromatic_trained.output/  `# The output directory` \
    --inference_label_sets test \
    --rotations 2 \
    --max_pdbs_test 100 \
    --json_params common_params.json.inp \
    | tee 8.OUT-python_out.txt
