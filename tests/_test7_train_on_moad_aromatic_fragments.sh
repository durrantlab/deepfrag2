echo "Train on a small subset of the Binding MOAD, only larger fragments that are aromatic"

mkdir -p 7.train_on_moad_large_aromatic_frags.output

python -u ../MainDF2.py \
    --mode train_on_moad \
    --max_epochs 3 \
    --save_params ./7.train_on_moad_large_aromatic_frags.output/params.saved.json \
    --save_splits ./7.train_on_moad_large_aromatic_frags.output/splits.saved.json \
    --csv moad/every.csv \
    --cache ./every_csv.cache.json \
    --data_dir moad/ \
    --default_root_dir $(pwd)/7.train_on_moad_large_aromatic_frags.output/  `# The output directory` \
    --json_params common_params.json.inp \
    --min_frag_num_heavy_atoms 4 \
    --mol_props aromatic \
    | tee 7.OUT-python_out.txt
