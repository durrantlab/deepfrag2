. _init.sh

echo "Perform inference using the fine-tuned model on a single example"

mkdir -p 5.inference.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode inference \
    --receptor ./data_for_inference/5VUP_prot_955.pdb \
    --branch_atm_loc_xyz "12.413000, 3.755000, 59.021999" \
    --ligand ./data_for_inference/5VUP_lig_955.sdf \
    --load_checkpoint ./3.finetune_moad.output/last.ckpt \
    --default_root_dir $(pwd)/5.inference.output/  `# The output directory` \
    --inference_label_sets ./data_for_inference/label_set.smi \
    --num_inference_predictions 10 \
    --rotations 20 \
    --json_params common_params.json.inp \
    | tee 4.OUT-python_out.txt
