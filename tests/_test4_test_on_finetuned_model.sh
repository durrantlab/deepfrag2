echo "Testing the finetuned data, using the test set withheld from the custom data"

mkdir -p 4.finetune_test.output

python -u ../MainDF2.py \
    --mode test_on_complexes \
    --load_splits ./3.finetune_custom.output/splits.saved.json \
    --load_checkpoint ./3.finetune_custom.output/last.ckpt \
    --csv ./data_to_finetune/pdb_sdf_file_pairs.csv \
    --data_dir ./data_to_finetune/ \
    --default_root_dir $(pwd)/4.finetune_test.output/  `# The output directory` \
    --inference_label_sets test \
    --rotations 2 \
    --json_params common_params.json.inp \
    --num_inference_predictions 5 \
    | tee 4.OUT-python_out.txt
