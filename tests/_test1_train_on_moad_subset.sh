echo "Train on a small subset of the Binding MOAD"

rm -rf 1.OUT-python_out.txt 1.train_on_moad.output
rm -rf moad/BindingMOAD_2020_mini/*pkl
mkdir -p 1.train_on_moad.output

python -u ../MainDF2.py \
    --mode train_on_moad \
    --max_epochs 3 \
    --save_params ./1.train_on_moad.output/params.saved.json \
    --save_splits ./1.train_on_moad.output/splits.saved.json \
    --csv moad/every.csv \
    --data_dir moad/ \
    --cache ./1.train_on_moad.output/every_csv.cache.json \
    --default_root_dir $(pwd)/1.train_on_moad.output/  `# The output directory` \
    --json_params common_params.json.inp \
    --cache_pdbs_to_disk \
    | tee 1.OUT-python_out.txt
