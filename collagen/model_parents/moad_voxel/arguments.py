"""Command-line arguments for the MOAD voxel model."""

from argparse import Namespace, ArgumentParser
from multiprocessing import cpu_count


def add_moad_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Add user-defined command-line parameters to control how the MOAD data is
    processed.
    
    Args:
        parent_parser (ArgumentParser): The parent parser to add the MOAD
            arguments to.
            
    Returns:
        ArgumentParser: The parser with the MOAD arguments added.
    """
    parser = parent_parser.add_argument_group("Binding MOAD")

    parser.add_argument(
        "--every_csv",
        required=False,
        help="Path to MOAD every.csv"
    )
    parser.add_argument(
        "--data_dir",
        required=False,  # Not required if running in --mode "inference"
        help="Path to MOAD root structure folder, or path to a folder containing a SDF file per each PDB file (protein-ligand pairs). This parameter can be used for both training and fine-tuning."
    )
    parser.add_argument(
        "--bad_data_dir",
        required=False,  # Not required if running in --mode "inference"
        help="Path to a folder containing bad ligands. This parameter is to be only used in the training mode as additional information to the data contained in --data_dir"
    )
    parser.add_argument(
        "--paired_data_csv",
        required=False,  # Not required if running in --mode "inference"
        help="This parameter is to be only used in the fine-tuning mode. This parameter is a set of comma separated values in the next order:\n"
             "1 - Path to the CSV file where information related to the paired data are stored\n"
             "2 - Column related to the PDB files\n"
             "3 - Column related to the SDF files\n"
             "4 - Column related to the parent SMILES strings\n"
             "5 - Column related to the SMILES strings of the first fragment\n"
             "6 - Column related to the SMILES strings of the second fragment\n"
             "7 - Column related to the activity value of the first fragment\n"
             "8 - Column related to the activity value of the second fragment\n"
             "9 - Column related to the receptor prevalence"
             "10 - Path to the PDB and SDF files"
    )

    # For many of these, good to define default values in args_defaults.py

    # NOTE: --custom_test_set_dir must be separate from --data_dir because you
    # might want to run a test on a given set of PDB files, but derive the label
    # sets from the BindingMOAD.
    parser.add_argument(
        "--custom_test_set_dir",
        required=False,
        # default=None,
        type=str,
        help="Path to a folder containing a SDF file per each PDB file (protein-ligand pairs). Used for testing on a user-specified directory of protein/ligand pairs."
    )
    parser.add_argument(
        "--fraction_train",
        required=False,
        # default=0.6,
        type=float,
        help="Percentage of targets to use in the TRAIN set."
    )
    parser.add_argument(
        "--fraction_val",
        required=False,
        # default=0.5,
        type=float,
        help="Percentage of (non-train) targets to use in the VAL set. The remaining ones will be used in the test set"
    )
    parser.add_argument(
        "--save_every_epoch",
        required=False,
        # default=False,
        action="store_true",
        help="To set if a checkpoint will be saved after finishing every training (or fine-tuning) epoch"
    )

    parser.add_argument(
        "--split_method",
        required=False,
        type=str,
        help="Method to use for splitting the data into TRAIN/VAL/TEST sets: (1) If 'random' (default), the data will be partitioned randomly according to the specified fractions. If any fragments are present in more than one set, some will be randomly removed to ensure independence. If 'high_priority', duplicate fragments will be removed so as to favor the training set first, then the validation set (i.e., training and validation sets will be larger that user-specified fraction). If 'low_priority', duplicate fragment will be removed so as to favor the test set, then the validation set (i.e., test and validation sets will be larger than the user-specified fraction). If 'butina', butina clustering will be used. Used only for finetuning. TODO: More details needed."
    )

    # Note that --prevent_smiles_overlap is no longer a user-definable
    # parameter.

    parser.add_argument(
        "--butina_cluster_cutoff",
        required=False,
        # default=None,
        type=float,
        help="Cutoff value to be applied for the Butina clustering method"
    )
    parser.add_argument(
        "--cache",
        required=False,
        # default=None,
        help="Path to MOAD cache.json file. If not given, `.cache.json` is appended to the file path given by `--every_csv`. If 'none' (default), will create new, temporary cache with a random filename.",
    )
    parser.add_argument(
        "--cache_pdbs_to_disk",
        # default=False,
        action="store_true",
        help="If given, collagen will convert the PDB files to a faster cachable format. Will run slower the first epoch, but faster on subsequent epochs and runs.",
    )
    parser.add_argument(
        "--noh",
        # default=True,
        action="store_true",
        help="If given, collagen will not use protein hydrogen atoms, nor will it save them to the cachable files generated with --cache_pdbs_to_disk. Can speed calculations and free disk space if your model doesn't need hydrogens, and if you're using --cache_pdbs_to_disk.",
    )
    parser.add_argument(
        "--discard_distant_atoms",
        # default=True,
        action="store_true",
        help="If given, collagen will not consider atoms that are far from any ligand, nor will it save them to the cachable files generated with --cache_pdbs_to_disk. Can speed calculations and free disk space if you're using --cache_pdbs_to_disk.",
    )
    parser.add_argument(
        "--split_seed",
        required=False,
        # default=1,
        type=int,
        help="Seed for TRAIN/VAL/TEST split. Defaults to 1.",
    )
    parser.add_argument(
        "--save_splits",
        required=False,
        # default=None,
        help="Path to a json file where the splits will be saved.",
    )
    parser.add_argument(
        "--load_splits",
        required=False,
        # default=None,
        type=str,
        help="Path to a json file (previously saved with --save_splits) describing the splits to use.",
    )
    parser.add_argument(
        "--max_pdbs_train",
        required=False,
        # default=None,
        type=int,
        help="If given, the max number of PDBs used to generate examples in the train set. If this set contains more than `max_pdbs_train` PDBs, extra PDBs will be removed.",
    )
    parser.add_argument(
        "--max_pdbs_val",
        required=False,
        # default=None,
        type=int,
        help="If given, the max number of PDBs used to generate examples in the val set. If this set contains more than `max_pdbs_val` PDBs, extra PDBs will be removed.",
    )
    parser.add_argument(
        "--max_pdbs_test",
        required=False,
        # default=None,
        type=int,
        help="If given, the max number of PDBs used to generate examples in the test set. If this set contains more than `max_pdbs_test` PDBs, extra PDBs will be removed.",
    )
    parser.add_argument(
        "--num_dataloader_workers",
        default=cpu_count(),
        type=int,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--max_voxels_in_memory",
        required=False,
        # default=512,
        type=int,
        help="The data loader will store no more than this number of voxel in memory at once.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        # default=16,
        help="The size of the batch. Defaults to 16.",
    )
    parser.add_argument(
        "--inference_limit",
        default=None,
        help="Maximum number of examples to run inference on. TODO: Not currently used.",
    )
    parser.add_argument(
        "--rotations",
        # default=8,
        type=int,
        help="Number of rotations to sample during inference or testing.",
    )
    parser.add_argument(
        "--inference_label_sets",
        default=None,
        type=str,
        help="A comma-separated list of the label sets to use during inference or testing. Does not impact DeepFrag training. If you are running DeepFrag in test mode, you must include the test set (for top-K metrics). Options: train, val, test, PATH to SMILES file. \n\nFor example, to include the val- and test-set compounds in the label set, as well as the compounds described in a file named `my_smiles.smi`: `val,test,my_smiles.smi`",
    )

    return parent_parser


def fix_moad_args(args: Namespace) -> Namespace:
    """Fix MOAD-specific arguments. Only works after arguments have been
    parsed, so in a separate definition.
    
    Args:
        args (Namespace): The arguments to fix.
        
    Returns:
        Namespace: The fixed arguments.
    """
    if args.cache is None:
        import os
        args.cache = f"{args.default_root_dir + os.sep}cache.json"
    elif args.cache == "none":
        # Note that this is now the default. Essentially, to recreate cache
        # every time (no cache from run to run, just within a run).
        import tempfile
        args.cache = tempfile.NamedTemporaryFile().name
    return args
