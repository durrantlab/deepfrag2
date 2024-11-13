
. _init.sh

echo "Testing the finetuned data, using the test set withheld from the custom data"

mkdir -p 4.finetune_test.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode test \
    --load_splits ./3.finetune_moad.output/splits.saved.json \
    --load_checkpoint ./3.finetune_moad.output/last.ckpt \
    --data_dir ./data_to_finetune/ \
    --default_root_dir $(pwd)/4.finetune_test.output/  `# The output directory` \
    --inference_label_sets test \
    --rotations 2 \
    --max_pdbs_test 100 \
    --json_params common_params.json.inp \
    --num_inference_predictions 5 \
    | tee 4.OUT-python_out.txt
