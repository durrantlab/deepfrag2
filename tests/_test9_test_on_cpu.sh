echo "Perform inference using the fine-tuned model on a single example"

mkdir -p 9.test_on_cpu.output

python -u ../MainDF2.py \
    --mode inference_single_complex \
    --receptor ./data_for_inference/5VUP_prot_955.pdb \
    --ligand ./data_for_inference/5VUP_lig_955.sdf \
    --branch_atm_loc_xyz 12.413000,3.755000,59.021999 \
    --default_root_dir $(pwd)/9.test_on_cpu.output/ \
    --rotations 20 \
    --load_checkpoint gte_4_best \
    --inference_label_sets ./data_for_inference/label_set.smi \
    --cache_pdbs_to_disk \
    --cache None \
    --cpu \
    --json_params common_params.json.inp \
    | tee 9.OUT-python_out.txt
