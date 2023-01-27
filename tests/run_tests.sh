echo "Loading modules and environment. Also, define PYTHON_EXEC, EVERY_CSV_BSNM (usually 'every.csv')"
echo "and MOAD_DIR variables. You must put all this in the env.sh file (not"
echo "synced to git)"

. env.sh

# Get python script path
MAIN_DF2_PY=`realpath $(ls ../MainDF2.py)`

echo "Train on a small subset of the Binding MOAD (for testing; --max_pdbs_train 100, --max_pdbs_val 100)"

mkdir -p 1.train_on_moad.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode train \
    --max_epochs 3 \
    --save_params ./1.train_on_moad.output/params.saved.json \
    --save_splits ./1.train_on_moad.output/splits.saved.json \
    --every_csv $MOAD_DIR/${EVERY_CSV_BSNM} \
    --cache $MOAD_DIR/${EVERY_CSV_BSNM}.cache.json \
    --data_dir $MOAD_DIR/ \
    --default_root_dir $(pwd)/1.train_on_moad.output/ \
    --max_pdbs_train 100 \
    --max_pdbs_val 100 \
    --json_params common_params.json.inp \
    | tee 1.OUT-python_out.txt

echo "Test on a small subset of the Binding MOAD (--max_pdbs_test 100)"

mkdir -p 2.test_moad_trained.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode test \
    --load_splits ./1.train_on_moad.output/splits.saved.json \
    --load_checkpoint ./1.train_on_moad.output/last.ckpt \
    --every_csv $MOAD_DIR/${EVERY_CSV_BSNM} \
    --cache $MOAD_DIR/${EVERY_CSV_BSNM}.cache.json \
    --data_dir $MOAD_DIR/ \
    --default_root_dir $(pwd)/2.test_moad_trained.output/ \
    --inference_label_sets test \
    --rotations 2 \
    --max_pdbs_test 100 \
    --json_params common_params.json.inp \
    | tee 2.OUT-python_out.txt


echo "Finetune on custom data"

mkdir -p 3.finetune_moad.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode warm_starting \
    --max_epochs 5 \
    --model_for_warm_starting ./1.train_on_moad.output/model_mean_mean_train.pt \
    --data_dir ./data_to_finetune/ \
    --default_root_dir $(pwd)/3.finetune_moad.output/ \
    --json_params common_params.json.inp \
    | tee 3.OUT-python_out.txt

echo "Perform inference using the fine-tuned model on a single example"

mkdir -p 4.inference.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode inference \
    --receptor ./data_for_inference/5VUP_prot_955.pdb \
    --branch_atm_loc_xyz "12.413000, 3.755000, 59.021999" \
    --ligand ./data_for_inference/5VUP_lig_955.sdf \
    --load_checkpoint ./3.finetune_moad.output/last.ckpt \
    --default_root_dir $(pwd)/4.inference.output/ \
    --inference_label_sets ./data_for_inference/label_set.smi \
    --num_inference_predictions 10 \
    --rotations 20 \
    --json_params common_params.json.inp \
    | tee 4.OUT-python_out.txt

echo "Perform inference using the fine-tuned model on a custom set"

mkdir -p 5.inference_custom_set.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --mode inference_custom_set \
    --default_root_dir $(pwd)/5.inference_custom_set.output/ \
    --load_checkpoint ./3.finetune_moad.output/last.ckpt \
    --custom_test_set_dir ./data_to_finetune/ \
    --every_csv $MOAD_DIR/${EVERY_CSV_BSNM} \
    --data_dir $MOAD_DIR/ \
    --inference_label_sets all \
    --rotations 8 \
    --json_params common_params.json.inp \
    | tee 5.OUT-python_out.txt
