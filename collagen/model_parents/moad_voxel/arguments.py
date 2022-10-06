from argparse import Namespace, ArgumentParser
from multiprocessing import cpu_count


def add_moad_args(parent_parser: ArgumentParser) -> ArgumentParser:
    # Add user-defined command-line parameters to control how the MOAD data is
    # processed.

    parser = parent_parser.add_argument_group("Binding MOAD")

    parser.add_argument(
        "--csv",
        required=False,
        default="/home/crg93/Data/crg93/moad.updated/every.csv",
        # default="D:\\Cesar\\0.Investigacion\\3.Experimentacion\\DeepFrag\\Datasets\\every.csv",
        help="Path to MOAD every.csv"
    )
    parser.add_argument(
        "--data",
        required=False,
        default="/home/crg93/Data/crg93/moad.updated/BindingMOAD_2020/",
        # default="D:\\Cesar\\0.Investigacion\\3.Experimentacion\\DeepFrag\\Datasets\\BindingMOAD_2020",
        help="Path to MOAD root structure folder"
    )
    parser.add_argument(
        "--cache",
        required=False,
        default=None,
        help="Path to MOAD cache.json file. If not given, `.cache.json` is appended to the file path given by `--csv`.",
    )
    parser.add_argument(
        "--cache_pdbs_to_disk",
        default=False,
        action="store_true",
        help="If given, collagen will convert the PDB files to a faster cachable format. Will run slower the first epoch, but faster on subsequent epochs and runs.",
    )
    parser.add_argument(
        "--noh",
        default=True,
        action="store_true",
        help="If given, collagen will not use protein hydrogen atoms, nor will it save them to the cachable files generated with --cache_pdbs_to_disk. Can speed calculations and free disk space if your model doesn't need hydrogens, and if you're using --cache_pdbs_to_disk.",
    )
    parser.add_argument(
        "--discard_distant_atoms",
        default=True,
        action="store_true",
        help="If given, collagen will not consider atoms that are far from any ligand, nor will it save them to the cachable files generated with --cache_pdbs_to_disk. Can speed calculations and free disk space if you're using --cache_pdbs_to_disk.",
    )
    parser.add_argument(
        "--split_seed",
        required=False,
        default=1,
        type=int,
        help="Seed for TRAIN/VAL/TEST split. Defaults to 1.",
    )
    parser.add_argument(
        "--save_splits",
        required=False,
        default=None,
        help="Path to a json file where the splits will be saved.",
    )
    parser.add_argument(
        "--load_splits",
        required=False,
        default=None,
        type=str,
        help="Path to a json file (previously saved with --save_splits) describing the splits to use.",
    )
    parser.add_argument(
        "--max_pdbs_train",
        required=False,
        default=None,
        type=int,
        help="If given, the max number of PDBs used to generate examples in the train set. If this set contains more than `max_pdbs_train` PDBs, extra PDBs will be removed.",
    )
    parser.add_argument(
        "--max_pdbs_val",
        required=False,
        default=None,
        type=int,
        help="If given, the max number of PDBs used to generate examples in the val set. If this set contains more than `max_pdbs_val` PDBs, extra PDBs will be removed.",
    )
    parser.add_argument(
        "--max_pdbs_test",
        required=False,
        default=None,
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
        default=512,
        type=int,
        help="The data loader will store no more than this number of voxel in memory at once.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=16,
        help="The size of the batch. Defaults to 16.",
    )
    parser.add_argument(
        "--inference_limit",
        default=None,
        help="Maximum number of examples to run inference on.",
    )
    parser.add_argument(
        "--inference_rotations",
        default=8,
        type=int,
        help="Number of rotations to sample during inference or testing.",
    )
    parser.add_argument(
        "--inference_label_sets",
        default="test",
        type=str,
        help="A comma-separated list of the label sets to use during inference or testing. If for testing, you must include the test set (for top-K metrics). Options: train, val, test, PATH to SMILES file.",
    )

    return parent_parser


def fix_moad_args(args: Namespace) -> Namespace:
    # Only works after arguments have been parsed, so in a separate definition.
    if args.cache is None:
        args.cache = f"{args.csv}.cache.json"
    return args
