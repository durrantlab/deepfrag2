echo "Test pip installation"

deepfrag2 \
    --receptor ./data_for_inference/5VUP_prot_955.pdb \
    --ligand ./data_for_inference/5VUP_lig_955.sdf \
    --branch_atm_loc_xyz 12.413000,3.755000,59.021999
