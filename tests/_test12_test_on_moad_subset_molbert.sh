echo "Test on a small subset of the Binding MOAD (--max_pdbs_test 100)"

mkdir -p 12.test_moad_trained_molbert.output

python -u ../MainDF2.py \
    --mode test_on_moad \
    --load_splits ./11.train_on_moad_molbert.output/splits.saved.json \
    --load_checkpoint ./11.train_on_moad_molbert.output/last.ckpt \
    --csv moad/every.csv \
    --data_dir moad/ \
    --cache ./11.train_on_moad_molbert.output/every_csv.cache.json \
    --default_root_dir $(pwd)/12.test_moad_trained_molbert.output/  `# The output directory` \
    --inference_label_sets test \
    --rotations 2 \
    --max_pdbs_test 100 \
    --json_params common_params.json.inp \
    --cache_pdbs_to_disk \
    --fragment_representation molbert | tee 12.OUT-python_out.txt
