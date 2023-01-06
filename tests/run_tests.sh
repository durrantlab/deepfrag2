echo "Loading modules and environment. Also, define PYTHON_EXEC,"
echo "and MOAD_DIR variables. You must put all this in the env.sh file (not"
echo "synced to git)"

. env.sh

# Get python script path
MAIN_DF2_PY=`realpath $(ls ../MainDF2.py)`

echo "Train on a small subset of the Binding MOAD (for testing; --max_pdbs_train 100, --max_pdbs_val 100)"

mkdir -p 1.train_on_moad.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --gpus 1 \
    --json_params 1.train_on_moad.json.inp \
    --every_csv   $MOAD_DIR/every.csv \
    --cache $MOAD_DIR/every.csv.cache.json \
    --data_dir  $MOAD_DIR/ \
    --default_root_dir $(pwd)/1.train_on_moad.output/ \
    --max_pdbs_train 100 \
    --max_pdbs_val 100 \
    | tee 1.OUT-python_out.txt


echo "Test on a small subset of the Bidning MOAD (--max_pdbs_test 100)"

mkdir -p 2.test_moad_trained.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --gpus 1 \
    --json_params 2.test_moad_trained.json.inp \
    --every_csv $MOAD_DIR/every.csv \
    --cache $MOAD_DIR/every.csv.cache.json \
    --data_dir $MOAD_DIR/ \
    --default_root_dir $(pwd)/2.test_moad_trained.output/ \
    --load_splits ./1.train_on_moad.output/splits.saved.json \
    --load_checkpoint ./1.train_on_moad.output/last.ckpt \
    --max_pdbs_test 100 \
    | tee 2.OUT-python_out.txt


echo "Finetune on custom data"

mkdir -p 3.finetune_moad.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --gpus 1 \
    --json_params 3.finetune_moad.json.inp \
    --data_dir ./data_to_finetune/ \
    --default_root_dir $(pwd)/3.finetune_moad.output/ \
    --model_for_warm_starting ./1.train_on_moad.output/model_mean_mean_train.pt \
    --verbose True \
    | tee 3.OUT-python_out.txt

echo "Perform inference on new data"

mkdir -p 4.inference.output

$PYTHON_EXEC -u $MAIN_DF2_PY \
    --gpus 1 \
    --json_params 4.inference.json.inp \
    --default_root_dir $(pwd)/4.inference.output/ \
    --load_checkpoint ./3.finetune_moad.output/last.ckpt \
    --receptor ./data_for_inference/5VUP_prot_955.pdb \
    --branch_atm_loc_xyz "12.413000, 3.755000, 59.021999" \
    --ligand ./data_for_inference/5VUP_lig_955.sdf \
    --num_inference_predictions 10 \
    | tee 4.OUT-python_out.txt

