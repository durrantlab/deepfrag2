# Bash script

. env.sh
. env2.sh

# finetuned_output_dir is first argument. Contains checkpoints.
finetuned_output_dir=$1

# initial_model_output_dir is second argument. Contains splits.
initial_model_output_dir=$2

# If the directory in finetuned_output_dir contains a subdirectory called
# "test_results", delete it.
if [ -d "./test_results" ]; then
    rm -r ./test_results
fi

# Create a new directory called "test_results" in the directory in
# finetuned_output_dir.
mkdir ./test_results

# Get all the files that start with "loss" in that directory.
# Sort them by the number after "loss=".
finetuned_ckpt_files=`ls $finetuned_output_dir/loss*.ckpt | sort -t= -k2 -n`

# Go through each of the files, and get the epoch number.
for finetuned_ckpt_file in $finetuned_ckpt_files; do
    # Get the epoch, which is the number after "epoch="
    finetuned_epoch=`echo $finetuned_ckpt_file | sed -e 's/.*epoch=\([0-9]*\).*/\1/'`

    # Create a subdirectory in the directory in finetuned_output_dir called
    # "test_results" called "epoch_finetuned_epoch".
    epoch_output_dir=./test_results/epoch_$finetuned_epoch/
    mkdir $epoch_output_dir

    # Create a subdirectories called finetuned and initial_model
    epoch_output_finetuned_dir=$epoch_output_dir/finetuned
    epoch_output_initial_model_dir=$epoch_output_dir/initial_model
    mkdir $epoch_output_finetuned_dir
    mkdir $epoch_output_initial_model_dir

    # First test the model on the finetuned data (expect accuracy to go up)
    $PYTHON_EXEC -u $MAIN_DF2_PY \
        --mode test \
        --load_splits ${finetuned_output_dir}/splits.saved.json \
        --load_checkpoint ${finetuned_ckpt_file} \
        --data_dir ../data_to_finetune/ \
        --default_root_dir ${epoch_output_finetuned_dir}  `# The output directory` \
        --inference_label_sets test \
        --rotations 2 \
        --json_params common_params.json.inp \
        | tee ${epoch_output_finetuned_dir}OUT-python_out.txt

    # Also test the model on the whole binding moad (expect accuracy to go down)
    $PYTHON_EXEC -u $MAIN_DF2_PY \
        --mode test \
        --load_splits ${initial_model_output_dir}/splits.saved.json \
        --load_checkpoint ${finetuned_ckpt_file}   `# Note loading finetuned checkpoint` \
        --every_csv $MOAD_DIR/${EVERY_CSV_BSNM} \
        --cache $MOAD_DIR/${EVERY_CSV_BSNM}.cache.json \
        --data_dir $MOAD_DIR/ \
        --default_root_dir ${epoch_output_initial_model_dir}  `# The output directory` \
        --inference_label_sets test \
        --rotations 2 \
        --json_params common_params.json.inp \
        | tee ${epoch_output_initial_model_dir}OUT-python_out.txt
done




# Get the epoch, which is the number after "epoch="
# epochs=`echo $finetuned_ckpt_files | sed -e 's/.*epoch=\([0-9]*\).*/\1/'`


# echo $finetuned_ckpt_files
# echo $epochs
