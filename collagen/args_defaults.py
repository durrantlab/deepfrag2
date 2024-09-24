"""In order to enable both command-line and API use, we need to define default
values for the arguments independently of the argument parser. This is done by
defining a function that returns a dictionary of default values. The function is
called by the argument parser to set the default values and by the API to get
the default values.

I'm only going to define defaults here for select arguments. The rest will be
defined by the argparser or must be explicitly defined via the API.
"""

# TODO: Defaults should be defined in their respective modules and "hooked"
# here.

from typing import Any, Dict
import argparse
from apps.deepfrag.AggregationOperators import Operator


def get_default_args() -> dict:
    """Return a dictionary of default arguments for the deepfrag app.

    Returns:
        dict: A dictionary of default arguments.
    """
    return {
        "fragment_representation": "rdk10",
        "aggregation_3x3_patches": Operator.MEAN.value,
        "aggregation_loss_vector": Operator.MEAN.value,
        "aggregation_rotations": Operator.MEAN.value,
        "cpu": False,
        "save_params": None,
        "learning_rate": 1e-4,
        "min_frag_mass": 0,
        "max_frag_mass": 150,
        "max_frag_dist_to_recep": 4,
        "min_frag_num_heavy_atoms": 1,
        "max_frag_num_heavy_atoms": 9999,
        "fraction_train": 0.6,
        "fraction_val": 0.5,
        "noh": True,
        "discard_distant_atoms": True,
        "split_seed": 1,
        "save_splits": None,
        "load_splits": None,
        "max_pdbs_train": None,
        "max_pdbs_val": None,
        "max_pdbs_test": None,
        "batch_size": 16,
        "rotations": 8,
        "cache_pdbs_to_disk": False,
        # "cache": "none",  # Means recreate cache every time. None would be append `.cache.json` to the file path given by `--every_csv
        "cache": None,
        "wandb_project": None,
        "save_every_epoch": False,
        "custom_test_set_dir": None,
        "every_csv": None,
        "split_method": "random_default",
        "butina_cluster_cutoff": None,
        "max_frag_repeats": None,
        "mol_props": "",
        "max_voxels_in_memory": 512,
    }


# Given a namespace, add any missing values
def add_missing_args_to_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Add missing arguments to the namespace.

    Args:
        args (argparse.Namespace): The existing arguments, to which the missing
            arguments will be added.

    Returns:
        argparse.Namespace: The updated namespace.
    """
    defaults = get_default_args()
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args
