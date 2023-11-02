# Create a function that detects if a file exists. If it does, it passes. Otherwise, it fails.
function test_file_exists {
    if [ -f $1 ]; then
        echo "PASS: $1 exists"
    else
        echo "FAIL: $1 does not exist"
    fi
}

# Now report results of tests
echo
echo "Results of tests"
echo "================"

test_file_exists ./1.train_on_moad.output/last.ckpt
test_file_exists ./2.test_moad_trained.output/predictions_MOAD/mean/test_results-1.json
test_file_exists ./3.finetune_moad.output/last.ckpt
test_file_exists ./4.finetune_test.output/predictions_nonMOAD/mean/test_results-1.json
test_file_exists ./5.inference.output/inference_out.smi
test_file_exists ./6.inference_custom_set.output/predictions_CustomDataset/mean/test_results-1.json
test_file_exists ./7.train_on_moad_large_aromatic_frags.output/last.ckpt
test_file_exists ./8.test_big_aromatic_trained.output/predictions_MOAD/mean/test_results-1.json