# DeepFrag2: A Deep Learning Framework for Fragment-Based Lead Optimization

## Overview

Lead optimization involves modifying ligands to improve specific properties such as binding affinity. We here present DeepFrag2, a convolutional neural network (CNN) that suggests optimizing fragment additions given the structure of a receptor/ligand complex. DeepFrag2 converts input receptor/parent complexes into 3D grids, where each grid point represents a cubic region of the 3D space (a voxel). We selected this representation because the 3D local context is important for fragment binding, and converting molecular structures to voxels allows us to apply CNNs, a network architecture that has been used successfully in computer vision. The DeepFrag2 output is a continuous-valued topological fingerprint of the suggested fragment to add. DeepFrag2 compares this output fingerprint to a database of fragments with precalculated fingerprints to recover the most suitable fragments for specific complexes.

We provide a helpful [DeepFrag2 Google Colab Notebook](https://colab.research.google.com/github/durrantlab/deepfrag2/blob/main/deepfrag2_colab_notebook.ipynb) for those who wish to try DeepFrag2 without installing any software. The notebook guides users through the process of choosing a receptor-ligand complex, selecting a branching point on the ligand, choosing a pre-trained DeepFrag2 model, and generating fragment suggestions. The results are displayed in an easy-to-read table and visual grid, allowing users to quickly assess the suggested fragments.

## Documentation

Below, the parameters to be used to run DeepFrag2 for training, testing and inference on external sets are described. The kind of fingerprint to be used to recover the most similar fragments is a common parameter to run DeepFrag2. In this version, it can be specified the `rdk10` and `molbert` values for the `fragment_representation` parameter. The next examples are described using `rdk10`.

The output directory after running the DeepFrag2 framework:

```text
DeepFrag
├── tb_logs                                     <- Dirctory containing the logs files.
├── predictions_MOAD                            <- Dirctory containing the results on the test set of the MOAD database.
├── predictions_nonMOAD                         <- Dirctory containing the results on the test set of a database other than MOAD database.
├── predictions_Single_Complex                  <- Dirctory containing the results of the inference process on a single receptor-ligand complex.
├── predictions_Multiple_Complexes              <- Dirctory containing the results of the inference process on multiples receptor-ligand complexes.
├── best.ckpt                                   <- Best trained model.
├── last.ckpt                                   <- Last trained model. This coincides with the best.ckpt file if all epochs are carried.
├── train-loss-epoch=XX-loss=YY.ckpt            <- One trained file per epoch, where XX is the epoch and YY is the loss value.
├── val-loss-epoch=XX-val_loss=YY.ckpt          <- One file per epoch for each trained model, where XX is the epoch and YY is the loss value.
├── cache.json                                  <- Cache file to be used when running the step mode to ensure comparability of results.
├── splits.json                                 <- File containing the training, validation, and test sets. this file will be used when running the step mode to ensure comparability of results.
├── train_on_XX.actually_used.json              <- File containing the chemical fragments of each ligand of a receptor-ligand complex that was used to train, XX is moad or custom if MOAD or custom database was used.
├── test_on_XX.actually_used.json               <- File containing the chemical fragments of each ligand of a receptor-ligand complex that was used in test mode, XX is moad or custom if MOAD or custom database was used.
├── model_train_last.pt                         <- .pt file of the trained model.
```

### Training on MOAD Database

```bash
python MainDF2.py \
    --csv path/to/moad/every.csv \
    --data_dir path/to/moad/BindingMOAD_2020 \
    --default_root_dir path/for/training/output \
    --max_epochs 60 \
    --fragment_representation rdk10 \
    --split_method random \
    --mode train_on_moad \
    --cache None \
    --cache_pdbs_to_disk \
    --gpus 1
```

#### Optional Parameters

```bash
--save_params path/to/save/the/training/output /training_parameters.json
--cpu [true if specified]
--save_every_epoch
--min_frag_num_heavy_atoms 1
--max_frag_num_heavy_atoms 9999
```

### Training on a Custom Database

To train on a custom database, use the same command line described above for training on the MOAD database, but set the `mode` parameter to `train_on_complexes`, the `--csv` parameter to a .csv file containing the paths of the receptor-ligand complexes, and the `--data_dir` parameter to the directory where the .pdb and .sdf files of the receptor-ligand complexes file are saved.

The input .csv file consists of two columns: one column named `receptor` that contains the paths of the .pdb files corresponding to the receptors, and another column named `ligand` that contains the paths of the .sdf files corresponding to the ligands.

### Testing on MOAD Database

To test on examples from the MOAD database, specify the paths of the `splits.json`, `cache.json`, and `model.ckpt` files via the `load_splits`, `cache`, and `load_checkpoint` parameters, respectively. These parameters should point to the training output directory since these files were created during the training process. Note that `model.ckpt` is a generic name used in this manual; the actual name of the .ckpt file should be specified in the `load_checkpoint` parameter.

```bash
python MainDF2.py \
    --csv path/to/moad/every.csv \
    --data_dir path/to/moad/BindingMOAD_2020 \
    --default_root_dir path/to/save/the/test/output \
    --fragment_representation rdk10 \
    --mode test_on_moad \
    --rotations 8 \
    --load_splits path/for/training/output/splits.json \
    --cache path/for/training/output/cache.json \
    --cache_pdbs_to_disk \
    --inference_label_sets test \
    --load_checkpoint path/for/training/output/model.ckpt \
    --gpus 1
```

#### Optional Parameters

```bash
--save_params path/to/save/the/training/output /test_parameters.json
--cpu [true if specified]
```

### Testing on a Custom Database

To test a trained DeepFrag2 model on a custom database, use the same command line used to test on the MOAD database, but set the `mode` parameter to `test_on_complexes`, the `--csv` parameter to a .csv file containing the paths of the receptor-ligand complexes, and the `--data_dir` parameter to the directory where the .pdb and .sdf files of the receptor-ligand complexes are located.

The input .csv file consists of two columns: one column named `receptor` that contains the paths of the .pdb files corresponding to the receptors, and another column named `ligand` that contains the paths of the .sdf files corresponding to the ligands.

### Inference on a Single Complex

This mode optionally allows you to include all chemical fragments from the MOAD database in the inference process by setting the `inference_label_sets` parameter to `all`.


```bash
python MainDF2.py \
    --csv path/to/moad/every.csv \
    --data_dir path/to/moad/BindingMOAD_2020 \
    --receptor path/to/the/receptor/file/receptor.pdb \
    --ligand path/to/the/ligand/file/ligand.sdf \
    --branch_atm_loc_xyz branching point 3D coordinates (e.g., 10.08,2.16,32.72) \
    --default_root_dir path/for/inference/output \
    --fragment_representation rdk10 \
    --mode inference_single_complex \
    --rotations 8 \
    --cache None \
    --cache_pdbs_to_disk \
    --inference_label_sets all \
    --load_checkpoint path/to/the/mode/model.ckpt \
    --gpus 1
```

In addition to the chemical fragments contained in the MOAD database, you can also provide one or several SMILES files containing different chemical fragments to be used as a label set during the inference process. Specfiy the SMILES files via the `inference_label_sets` parameter.

```bash
python MainDF2.py \
    --csv path/to/moad/every.csv \
    --data_dir path/to/moad/BindingMOAD_2020 \
    --receptor path/to/the/receptor/file/receptor.pdb \
    --ligand path/to/the/ligand/file/ligand.sdf \
    --branch_atm_loc_xyz branching point 3D coordinates (e.g., 10.08,2.16,32.72) \
    --default_root_dir path/for/inference/output \
    --fragment_representation rdk10 \
    --mode inference_single_complex \
    --rotations 8 \
    --cache None \
    --cache_pdbs_to_disk \
    --inference_label_sets all,path/to/file1.smiles,path/to/file2.smiles \
    --load_checkpoint path/to/the/mode/model.ckpt \
    --gpus 1
```

Users are not required to include the fragments of the MOAD database in the label set during inference. That is, you can consider only the chemical fragments provided in the SMILES files.

```bash
python MainDF2.py \
    --receptor path/to/the/receptor/file/receptor.pdb \
    --ligand path/to/the/ligand/file/ligand.sdf \
    --branch_atm_loc_xyz branching point 3D coordinates (e.g., 10.08,2.16,32.72) \
    --default_root_dir path/for/inference/output \
    --fragment_representation rdk10 \
    --mode inference_single_complex \
    --rotations 8 \
    --cache None \
    --cache_pdbs_to_disk \
    --inference_label_sets path/to/file1.smiles,path/to/file2.smiles \
    --load_checkpoint path/to/the/mode/model.ckpt \
    --gpus 1
```

#### Optional Parameters

```bash
--save_params path/to/save/the/training/output /inference_parameters.json
--cpu [true if specified]
--min_frag_num_heavy_atoms 1
--max_frag_num_heavy_atoms 9999
```

### Inference on Multiple Complexes

To run inference on multiple complexes, you can optionally use all fragments derived from the MOAD database as a label set; in this case, the `inference_label_sets` parameter should equal the value `all`.

Users should specify the multiple receptor/ligand complexes in a .csv file that contains two columns: one column named `receptor` has all paths to the PDB files, and another column named `ligand` has all paths of the SDF files.

```bash
python MainDF2.py \
    --csv path/to/moad/every.csv \
    --data_dir path/to/moad/BindingMOAD_2020 \
    --csv_complexes path/to/csv/file/describing/receptor-ligand-complexes/file.csv \
    --path_complexes path/containing/pdb-sdf-files/specified/in/csv_complexes/ \
    --default_root_dir path/to/save/the/inference/output \
    --fragment_representation rdk10 \
    --mode inference_multiple_complexes \
    --rotations 8 \
    --cache None \
    --cache_pdbs_to_disk \
    --inference_label_sets all \
    --load_checkpoint path/to/the/mode/model.ckpt \
    --gpus 1
```

### Using Already-Trained DeepFrag2 Models for Inference

To run DeepFrag2 in inference mode, users must specify the path of the .ckpt file corresponding to a specific DeepFrag2 model. The names of already-trained models (trained on the Binding MOAD database) can also be specified in the `--load_checkpoint` parameter, instead of the path to a specific model. These in-house models will be automatically downloaded into a directory  named `in-house_models`, which is created in the main directory of the DeepFrag2 framework. The name of the in-house models are given below:

| Name                     | Description                                                                                                     |
|--------------------------|-----------------------------------------------------------------------------------------------------------------|
| all_best                 | Model trained on the entire MOAD database for all chemical fragment sizes                                       |
| gte_4_acid_best          | Model trained on the MOAD database only considering acid chemical fragments with at least four heavy atoms      |
| gte_4_aliphatic_best     | Model trained on the MOAD database only considering aliphatic chemical fragments with at least four heavy atoms |
| gte_4_aromatic_best      | Model trained on the MOAD database only considering aromatic chemical fragments with at least four heavy atoms  |
| gte_4_base_best          | Model trained on the MOAD database only considering base chemical fragments with at least four heavy atoms      |
| gte_4_best               | Model trained on the MOAD database considering all chemical fragments with at least four heavy atoms            |
| lte_3_best               | Model trained on the MOAD database considering all chemical fragments containing as maximum three heavy atoms   |

### Reusing Calculated Fingerprints

Users can also specify fragment SMILES files to use as a custom label set via the `inference_label_sets` parameter. This option allows DeepFrag2 to consider chemical fragments that are not included in the small molecules of the MOAD database.

When running DeepFrag2 in any inference mode, the fingerprints calculated for the chemical fragments are automatically saved (cached) to local files to avoid recomputing the same fingerprints for the same chemical fragments. Users must specify the same paths to the MOAD database and/or SMILES files to reuse the .pt files containing the calculated fingerprints.

### Fingerprints

In the examples above, we used the `rdk10` fingerprint representation. You can also specify two other types of fingerprint representations. The first is a combination of `rdk10` and `morgan` fingerprints named `rdk10_x_morgan`. The other is named `molbert`, which uses the `molbert` large language model that is freely available at [https://github.com/BenevolentAI/MolBERT](https://github.com/BenevolentAI/MolBERT).

To use these fingerprint representations, change the `fragment_representation` parameter in the command line examples above.
