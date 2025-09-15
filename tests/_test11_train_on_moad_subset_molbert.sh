echo "Train on a small subset of the Binding MOAD"

rm -rf 11.OUT-python_out.txt 11.train_on_moad_molbert.output
rm -rf moad/BindingMOAD_2020_mini/*pkl
mkdir -p 11.train_on_moad_molbert.output

python -u ../MainDF2.py \
    --mode train_on_moad \
    --max_epochs 3 \
    --save_params ./11.train_on_moad_molbert.output/params.saved.json \
    --save_splits ./11.train_on_moad_molbert.output/splits.saved.json \
    --csv moad/every.csv \
    --data_dir moad/ \
    --cache ./11.train_on_moad_molbert.output/every_csv.cache.json \
    --default_root_dir $(pwd)/11.train_on_moad_molbert.output/  `# The output directory` \
    --json_params common_params.json.inp \
    --cache_pdbs_to_disk \
    --fragment_representation molbert | tee 11.OUT-python_out.txt
