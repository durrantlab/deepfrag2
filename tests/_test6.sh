. _init.sh

echo "Perform inference using the fine-tuned model on a custom set"

mkdir -p 6.inference_custom_set.output

# NOTE TO JDD: THIS IS NOT TO TEST A FINETUNED MODEL. IT'S TO APPLY A TRAINED
# FINETUNED MODEL TO A NEW SET OF COMPLEXES. TO TEST A FINETUNED MODEL, JUST USE
# --MODE TEST, AS ABOVE.

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode inference_custom_set \
    --default_root_dir $(pwd)/6.inference_custom_set.output/  `# The output directory` \
    --load_checkpoint ./3.finetune_moad.output/last.ckpt \
    --custom_test_set_dir ./data_to_finetune/  `# separate from --data_dir so you can run test on one PDB set, but get labels from BindingMOAD.` \
    --every_csv $MOAD_DIR/${EVERY_CSV_BSNM} \
    --data_dir $MOAD_DIR/  `# For labels` \
    --inference_label_sets all \
    --rotations 2 \
    --json_params common_params.json.inp \
    | tee 5.OUT-python_out.txt
