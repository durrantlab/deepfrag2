
# Overview

Lead optimization involves modifying ligands to improve specific properties such as binding affinity. We here present 
DeepFrag, a convolutional neural network (CNN) that suggests optimizing fragment additions given the structure of a 
/ligand complex. DeepFrag converts input receptor/parent complexes into 3D grids, where each grid point represents a 
cubic region of the 3D space (a voxel). We selected this representation because the 3D local context is important for 
fragment binding, and converting molecular structures to voxels allows us to apply CNNs, a network architecture that 
has been used successfully in computer vision. The DeepFrag output is a continuous-valued topological fingerprint of 
the suggested fragment to add. DeepFrag compares this output fingerprint to a database of fragments with precalculated
fingerprints to recover the most suitable fragments for specific ligands. 

# Documentation

Below, the parameters to be used to run DeepFrag for training, testing and inference on external sets are described.
The kind of fingerprint to be used to recover the most similar fragments is a common parameter tu run DeepFrag. In this
version, it can be specified the 'rdk10' and 'molbert' values for the 'fragment_representation' parameter. The next 
examples are described using 'rdk10'.

The output directory after running the DeepFrag framework:
```
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

## **For training step on MOAD database**
```
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
### Optional parameters
```
--save_params path/to/save/the/training/output /training_parameters.json
--cpu True (to use CPU and not GPU)
--save_every_epoch
--min_frag_num_heavy_atoms 1
--max_frag_num_heavy_atoms 9999
```

## **For training step on a custom database**
To this end, it is used the same command line described for the training process on the MOAD database. But in this case, 
the 'mode' parameter should be equal to 'train_on_complexes', the '--csv' parameter should be pointed out to a .csv file 
containing the path of the receptor-ligand complexes; and the '--data_dir' parameter should be pointed out to the 
directory where the .pdb and .sdf files of the receptor-ligand complexes described in the input .csv file are saved. 

The input .csv file is comprised of two columns, one column named 'receptor' that contains the path of the .pdb files 
corresponding to the receptors, while the other column, named 'ligand', contains the path of the .sdf files 
corresponding to the ligands.

## **For test step for MOAD database**

In this mode, the path of the ‘split.json’, ‘cache.json’, and ‘model.ckpt’ files specified in the ‘load_splits’, 
‘cache’, and ‘load_checkpoint’ parameters, respectively, should be pointed to the directory of the training output
since these files were created during the training process. Noticed that ‘model.ckpt’ is a generic name used in this 
manual and, thus, the true name of the .ckpt file should be specified in the ‘load_checkpoint’ parameter.
```
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
### Optional parameters
```
--save_params path/to/save/the/training/output /test_parameters.json
--cpu True (to use CPU and not GPU)
```
## **For test step on a custom database**
To this end, it can be used the same command line to run the test mode on the MOAD database. But in case, the 'mode' 
parameter should be equal to 'train_on_complexes', the '--csv' parameter should be pointed out to a .csv file containing 
the path of the receptor-ligand complexes, and the '--data_dir' parameter should be pointed out to the directory where 
the .pdb and .sdf files of the receptor-ligand complexes described in the input .csv file are saved. 

The input .csv file is comprised of two columns, one column named 'receptor' that contains the path of the .pdb files 
corresponding to the receptors, while the other column, named 'ligand', contains the path of the .sdf files 
corresponding to the ligands.

## **For inference step on a single complex**

In this mode, the MOAD database (optional) could be specified to consider all their chemical fragments in the inference 
process. If specified, the ‘inference_label_sets’ parameter should contain the value ‘all’.
```
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
In addition to the chemical fragment contained in the MOAD database, you could also provide one or several SMILES files 
containing different chemical fragments to be used in the inference process. The SMILES files are specified in the 
‘inference_label_sets’ parameter.
```
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
As mentioned before, it is not mandatory to specify the MOAD database to consider their chemical fragments for 
inference. That is, it can be considered only chemical fragments provided in the SMILES files.
```
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
### Optional parameters
```
--save_params path/to/save/the/training/output /inference_parameters.json
--cpu True (to use CPU and not GPU)
--min_frag_num_heavy_atoms 1
--max_frag_num_heavy_atoms 9999
```
## **For inference step on multiple complexes**

In this mode, the MOAD database (optional) could be specified to consider all their chemical fragments in the inference 
step. If specified, the ‘inference_label_sets’ parameter should contain the value ‘all’.

The difference between the inference process on a single complex and multiple complexes lies on the former is only 
carried out for a single receptor/ligand complex; whereas the latter is carried out on several complex/ligand complexes
that are described in a .csv file containing two columns.  One column named ‘receptor’ with all paths of the PDB files, 
and the other column named ligand with all paths of the SDF files.
```
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
Similar to the inference process on a single receptor-ligand complex, SMILES files can be specified in the 
‘inference_label_sets’ parameter to consider chemical fragments other than the ones contained in the MOAD database.

## **Reusing calculated fingerprints**

When running the inference modes, the fingerprints calculated for the chemical fragments into the MOAD database and/or
into the SMILES files are automatically saved in the same directories where they are. These files are saved to avoid 
recomputing the same fingerprints for the same chemical fragments. Thus, it is necessary to specify the same directories
where the MOAD database and/or SMILES files were used for inference to reuse the .pt files containing the calculated 
fingerprints.

## **Fingerprints**

The previous examples were using the 'rdk10' fingerprint representation. Additionally, it can be specified other two 
types of fingerprint representations. One of them is a combination between 'rdk10' and 'morgan' fingerprints named as
'rdk10_x_morgan'. The another one is named as 'molbert'. For its calculation, it is used the 'molbert' large language 
model freely available at https://github.com/BenevolentAI/MolBERT.

To use these fingerprint representations, it should be modified the 'fragment_representation' parameter in the command
line explained above.
