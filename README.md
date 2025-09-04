# DeepFrag2: A Deep Learning Framework for Fragment-Based Lead Optimization

## Overview

Lead optimization involves modifying ligands to improve specific properties such as binding affinity. We here present DeepFrag2, a convolutional neural network (CNN) that suggests optimizing fragment additions given the structure of a receptor/ligand complex.

DeepFrag2 converts input receptor/parent complexes into 3D grids, where each grid point represents a cubic region of the 3D space (a voxel). We selected this representation because the 3D local context is important for fragment binding, and converting molecular structures to voxels allows us to apply CNNs, a network architecture that has been used successfully in computer vision. The DeepFrag2 output is a continuous-valued topological fingerprint of the suggested fragment to add. DeepFrag2 compares this output fingerprint to a database of fragments with precalculated fingerprints to recover the most suitable fragments for specific complexes.

### Try DeepFrag2 in Your Browser

We provide a helpful **[DeepFrag2 Google Colab Notebook](https://colab.research.google.com/github/durrantlab/deepfrag2/blob/main/deepfrag2_colab_notebook.ipynb)** for those who wish to try DeepFrag2 without installing any software. The notebook guides users through the process of choosing a receptor-ligand complex, selecting a branching point on the ligand, choosing a pre-trained DeepFrag2 model, and generating fragment suggestions. The results are displayed in an easy-to-read table and visual grid, allowing users to quickly assess the suggested fragments.

## Installation

For detailed instructions on setting up the Conda environment and configuring the project, please see the **[Installation Guide](./INSTALL_NOTES.md)**. We provide two Conda environments:

* **GPU Environment**: Required for training new models.
* **CPU-Only Environment**: A lightweight option that is easy to install (no CUDA required) and very fast for running inference with pre-trained models. For most users who wish only to use DeepFrag2, this is the preferred option.

## Usage

This section describes the parameters for running DeepFrag2 for training, testing, and inference. The examples below use the `rdk10` fingerprint representation, but other options are available (see the "Fingerprints" section).

### Output Directory Structure

After running, the output directory (`--default_root_dir`) will be organized as follows:

```text
.
├── tb_logs/                     <- Directory for TensorBoard log files.
├── predictions_MOAD/            <- Results from testing on the MOAD database.
├── predictions_nonMOAD/         <- Results from testing on a custom database.
├── predictions_Single_Complex/  <- Inference results for a single complex.
├── predictions_Multiple_Complexes/<- Inference results for multiple complexes.
├── best.ckpt                    <- Checkpoint for the model with the best validation loss.
├── last.ckpt                    <- The most recent checkpoint from the last training epoch.
├── val-loss-epoch=XX-val_loss=YY.ckpt <- Checkpoint from each validation epoch.
├── train-loss-epoch=XX-loss=YY.ckpt <- Checkpoint from each training epoch (if saved).
├── cache.json                   <- Cached molecular properties to speed up data loading.
├── splits.json                  <- Contains the train, validation, and test set splits.
├── *.actually_used.json         <- Lists fragments from each set used in training/testing.
└── model_train_last.pt          <- The final trained model state dictionary.
```

### Training on MOAD Database

```bash
python MainDF2.py \
    --mode train_on_moad \
    --csv path/to/moad/every.csv \
    --data_dir path/to/moad/BindingMOAD_2020 \
    --default_root_dir path/for/training/output \
    --fragment_representation rdk10 \
    --max_epochs 60 \
    --split_method random \
    --cache None \
    --cache_pdbs_to_disk \
    --gpus 1
```

#### Optional Training Parameters

```bash
--save_params path/to/save/parameters.json  # Save all parameters to a JSON file.
--cpu                                       # Force CPU usage.
--save_every_epoch                          # Save a checkpoint after every epoch.
--min_frag_num_heavy_atoms 1                # Minimum heavy atoms in a fragment.
--max_frag_num_heavy_atoms 9999             # Maximum heavy atoms in a fragment.
```

### Training on a Custom Database

To train on a custom database, set `--mode` to `train_on_complexes`. You must also provide:

- A `--csv` file that lists the receptor-ligand pairs. This file must contain two columns: `receptor` (for PDB files) and `ligand` (for SDF files).

Example:

```csv
receptor,ligand
3JT8_prot_165.pdb,3JT8_lig_165.sdf
3JTA_prot_172.pdb,3JTA_lig_172.sdf
3NLO_prot_287.pdb,3NLO_lig_287.sdf
3UFP_prot_383.pdb,3UFP_lig_383.sdf
3UFW_prot_411.pdb,3UFW_lig_411.sdf
4CTR_prot_449.pdb,4CTR_lig_449.sdf
4CTR_prot_450.pdb,4CTR_lig_450.sdf
4JSJ_prot_585.pdb,4JSJ_lig_585.sdf
5FVT_prot_820.pdb,5FVT_lig_820.sdf
5VUY_prot_991.pdb,5VUY_lig_991.sdf
```

- The `--data_dir` parameter, which should point to the base directory where the PDB and SDF files are located.

### Testing on MOAD Database

To test a model, you must provide the `splits.json`, `cache.json`, and model checkpoint (`.ckpt`) files generated during training.

```bash
python MainDF2.py \
    --mode test_on_moad \
    --csv path/to/moad/every.csv \
    --data_dir path/to/moad/BindingMOAD_2020 \
    --default_root_dir path/to/save/test/output \
    --load_checkpoint path/for/training/output/best.ckpt \
    --load_splits path/for/training/output/splits.json \
    --cache path/for/training/output/cache.json \
    --cache_pdbs_to_disk \
    --fragment_representation rdk10 \
    --inference_label_sets test \
    --rotations 8 \
    --gpus 1
```

#### Optional Testing Parameters

```bash
--save_params path/to/save/test_parameters.json
--cpu
```

### Testing on a Custom Database

To test on a custom database, set the `--mode` to `test_on_complexes` and provide the `--csv` and `--data_dir` parameters pointing to your custom dataset. The input `.csv` file requires the same format described for custom training: one column named `receptor` for PDB files and one named `ligand` for SDF files.

### Inference on a Single Complex

To get fragment suggestions for a single receptor-ligand pair, use `inference_single_complex` mode. You must specify the receptor, ligand, and the 3D coordinates of the branching atom.

```bash
python MainDF2.py \
    --mode inference_single_complex \
    --receptor path/to/the/receptor.pdb \
    --ligand path/to/the/ligand.sdf \
    --branch_atm_loc_xyz "10.08,2.16,32.72" \
    --default_root_dir path/for/inference/output \
    --load_checkpoint path/for/training/output/best.ckpt \
    --fragment_representation rdk10 \
    --inference_label_sets all,path/to/my_fragments.smiles \
    --csv path/to/moad/every.csv \
    --data_dir path/to/moad/BindingMOAD_2020 \
    --rotations 8 \
    --gpus 1
```

The `--inference_label_sets` parameter defines the library of fragments to search for suggestions. It accepts a comma-separated list:
-   `all`: Includes all fragments from the BindingMOAD database. Requires `--csv` and `--data_dir` to be set.
-   `path/to/file.smiles`: Includes fragments from a custom SMILES file.
-   You can combine them, e.g., `all,path/to/file1.smiles,path/to/file2.smiles`.
-   If `all` is omitted, only fragments from the provided SMILES files are used.

#### Optional Inference Parameters

```bash
--save_params path/to/save/inference_parameters.json
--cpu
--min_frag_num_heavy_atoms 1
--max_frag_num_heavy_atoms 9999
```

### Inference on Multiple Complexes

To run inference on a batch of complexes, use `inference_multiple_complexes`. The `--csv_complexes` file should contain `receptor` and `ligand` columns with file paths.

```bash
python MainDF2.py \
    --mode inference_multiple_complexes \
    --csv_complexes path/to/your/complexes.csv \
    --path_complexes path/containing/your/pdb_sdf_files/ \
    --default_root_dir path/to/save/inference/output \
    --load_checkpoint path/for/training/output/best.ckpt \
    --fragment_representation rdk10 \
    --inference_label_sets all \
    --csv path/to/moad/every.csv \
    --data_dir path/to/moad/BindingMOAD_2020 \
    --rotations 8 \
    --gpus 1
```

### Using Pre-trained Models for Inference

You can use our pre-trained models by specifying their name with `--load_checkpoint` instead of a file path. The models will be downloaded automatically into an `in-house_models` directory.

| Name                     | Description                                                                                                     |
|--------------------------|-----------------------------------------------------------------------------------------------------------------|
| `all_best`                 | Model trained on the entire MOAD database for all chemical fragment sizes.                                      |
| `gte_4_acid_best`          | Trained on acid fragments with at least four heavy atoms.                                                       |
| `gte_4_aliphatic_best`     | Trained on aliphatic fragments with at least four heavy atoms.                                                  |
| `gte_4_aromatic_best`      | Trained on aromatic fragments with at least four heavy atoms.                                                   |
| `gte_4_base_best`          | Trained on base fragments with at least four heavy atoms.                                                       |
| `gte_4_best`               | Trained on all fragments with at least four heavy atoms.                                                        |
| `lte_3_best`               | Trained on all fragments with a maximum of three heavy atoms.                                                   |

### Reusing Calculated Fingerprints

When running inference with `--inference_label_sets all`, DeepFrag2 automatically caches the calculated fingerprints of fragments to speed up subsequent runs. These cache files (`*_all_label_set_fps.bin` and `*_all_label_set_smis.bin`) are saved in the same directory as the MOAD `every.csv` file specified by the `--csv` parameter. To clear the cache and force regeneration, you must delete these `.bin` files.

### Fingerprints

DeepFrag2 supports several fingerprint representations, specified with the `--fragment_representation` flag.

-   `rdk10` (Default): A topological fingerprint from RDKit.
-   `rdk10_x_morgan`: A combination of RDKit and Morgan fingerprints.
-   `molbert`: Uses the [MolBERT](https://github.com/BenevolentAI/MolBERT) large language model.

To use a different fingerprint, simply change the value of the `--fragment_representation` parameter in the command line examples.
