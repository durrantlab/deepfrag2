. _init.sh

echo "Inference on a single complex by giving the receptor, the ligand, and the connection point"

mkdir -p 9.inference_single_complex.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode inference_single_complex \
    --receptor ./data_for_inference/5VUP_prot_955.pdb \
    --ligand ./data_for_inference/5VUP_lig_955.sdf \
    --branch_atm_loc_xyz 12.413000,3.755000,59.021999 \
    --default_root_dir $(pwd)/9.inference_single_complex.output/ \
    --rotations 2 \
    --aggregation_rotations mean \
    --load_checkpoint ./1.train_on_moad.output/last.ckpt \
    --inference_label_sets ./data_for_inference/label_set.smi \
    --fragment_representation rdk10 \
    --cache_pdbs_to_disk \
    --cache None \
    | tee 9.OUT-python_out.txt
