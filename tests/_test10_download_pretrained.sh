echo "Perform inference using pretrained model and label set on a custom set"

mkdir -p 10.pretrained_inference_custom_set.output

# NOTE TO JDD: THIS IS NOT TO TEST A FINETUNED MODEL. IT'S TO APPLY A TRAINED
# FINETUNED MODEL TO A NEW SET OF COMPLEXES. 

python -u ../MainDF2.py \
    --mode inference_multiple_complexes \
    --csv_complexes ./data_for_inference/pdb_sdf_file_pairs.csv \
    --path_complexes ./data_for_inference/ \
    --default_root_dir $(pwd)/10.pretrained_inference_custom_set.output/ \
    --rotations 2 \
    --load_checkpoint gte_4_best \
    --inference_label_sets gte_4_all \
    --cache_pdbs_to_disk \
    --cache None \
    | tee 10.OUT-python_out.txt
