rm -f data_for_inference/*.bin
rm -rf moad/BindingMOAD_2020_mini/*pkl
rm -f ?.OUT-*
rm -rf ?.*.output
rm -f ./every_csv.cache.json
rm -f tmp.tsv
rm -f bad_smiles.log

# Ask for confirmation before deleting pretrained_models
if [ -d "pretrained_models" ]; then
    read -p "Delete pretrained_models directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing pretrained_models..."
        rm -rf pretrained_models
    else
        echo "Skipping pretrained_models deletion."
    fi
else
    echo "pretrained_models directory not found."
fi
