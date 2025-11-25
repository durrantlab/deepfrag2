"""Creates an untrained DeepFrag model checkpoint for baseline testing.

This script initializes the model architecture with random weights (standard
PyTorch initialization) and saves it to a .ckpt file. This file can then be
passed to test.py or inference scripts to establish a random baseline.
"""

import os
import torch
from collagen.core.args import get_args
from collagen.apps.deepfrag.model import DeepFragModel
from collagen.apps.deepfrag.model_fusing_modalities import DeepFragModelESM2
from collagen.model_parents.moad_voxel.moad_voxel import VoxelModelParent
from collagen.external.common.datasets.fragment_dataset import FragmentDataset
from collagen.core.voxelization.voxelizer import VoxelParamsDefault


def main():
    """Entry point for the script."""
    # Parse arguments exactly like the main training script to ensure
    # architecture hyperparameters (like hidden sizes) match your real models.
    args = get_args(
        parser_funcs=[
            VoxelModelParent.add_moad_args,
            DeepFragModel.add_model_args,
            DeepFragModelESM2.add_model_args,
            FragmentDataset.add_fragment_args,
        ],
        post_parse_args_funcs=[VoxelModelParent.fix_moad_args],
        is_pytorch_lightning=True,
    )

    print("Initializing model with random weights...")

    # Calculate input features based on default voxel parameters used in main.py
    voxel_params = VoxelParamsDefault.DeepFrag
    num_voxel_features = (
        voxel_params.receptor_featurizer.size() +
        voxel_params.ligand_featurizer.size()
    )

    # VoxelModelParent.setup_fingerprint_scheme sets args.fp_size based on
    # args.fragment_representation. This is required by the model __init__.
    VoxelModelParent.setup_fingerprint_scheme(args)

    # Determine if using multimodal or standard model
    if hasattr(args, "run_mm_model") and args.run_mm_model:
        # Note: This might download ESM models if not present
        model = DeepFragModelESM2(
            num_voxel_features=num_voxel_features,
            **vars(args)
        )
    else:
        model = DeepFragModel(
            num_voxel_features=num_voxel_features,
            **vars(args)
        )

    # Create checkpoint dictionary mimicking PyTorch Lightning structure
    # so it can be loaded by VoxelModelParent.load_from_checkpoint
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": vars(args),
        "epoch": 0,
        "global_step": 0,
    }

    output_path = "random_untrained.ckpt"
    abs_path = os.path.abspath(output_path)

    torch.save(checkpoint, output_path)

    print(f"Saved untrained checkpoint to: {abs_path}")
    print("\nTo test this baseline, run your standard test command, e.g.:")
    print(f"  python apps/deepfrag/run.py --mode test_on_moad "
          f"--load_checkpoint {output_path} ...")


if __name__ == "__main__":
    main()