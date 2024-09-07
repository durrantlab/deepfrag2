"""A model for inference on a custom dataset."""

from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.types import StructureEntry, StructuresSplit
from collagen.external.pdb_sdf_dir.interface import PdbSdfDirInterface
from collagen.model_parents.moad_voxel.test import VoxelModelTest
from argparse import ArgumentParser, Namespace
import os
from collagen.model_parents.moad_voxel.test_inference_utils import (
    remove_redundant_fingerprints,
)
import torch  # type: ignore
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from collagen.core.voxelization.voxelizer import VoxelParams
import pickle


class VoxelModelInferenceCustomSet(VoxelModelTest):

    """A model for inference on a custom set."""

    def __init__(self, model_parent: Any):
        """Initialize the model.

        Args:
            model_parent (Any): The parent model.
        """
        VoxelModelTest.__init__(self, model_parent)

    def _create_label_set(
        self,
        args: Namespace,
        device: torch.device,
        data_interface: ParentInterface,
        voxel_params: VoxelParams,
        existing_label_set_fps: torch.Tensor = None,
        existing_label_set_entry_infos: Optional[List[StructureEntry]] = None,
        skip_test_set=False,
        train_split: Optional[StructuresSplit] = None,
        val_split: Optional[StructuresSplit] = None,
        test_split: Optional[StructuresSplit] = None,
        lbl_set_codes: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Create a label set (look-up) tensor and smiles list for inference
        on custom label set. It can be comprised of the fingerprints in the
        BindingMOAD database, as well as SMILES strings from a file.

        Args:
            self: This object
            args (Namespace): The user arguments.
            device (torch.device): The device to use.
            data_interface (ParentInterface, optional): The dataset interface.
                Defaults to None.
            voxel_params (VoxelParams): Parameters for voxelization. Defaults
                to None.
            existing_label_set_fps (torch.Tensor, optional): The existing tensor
                of fingerprints to which these new ones should be added.
                Defaults to None.
            existing_label_set_entry_infos (List[StructureEntry], optional):
                Infos about any existing label set entries to which these new
                ones should be added. Defaults to None.
            skip_test_set (bool, optional): Do not add test-set fingerprints,
                presumably because they are already present in
                existing_label_set_entry_infos. Defaults to False.
            train (StructuresSplit, optional): The train split. Defaults to None.
            val (StructuresSplit, optional): The val split. Defaults to None.
            test (StructuresSplit, optional): The test split. Defaults to None.
            lbl_set_codes (List[str], optional): The list of label set codes.
                Comes from inference_label_sets. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[str]]: The updated fingerprint
                tensor and smiles list.
        """
        if (
            "train" in args.inference_label_sets
            or "val" in args.inference_label_sets
            or "test" in args.inference_label_sets
        ):
            raise Exception(
                "The 'all' value or SMILES strings are the only values allowed for the --inference_label_sets parameter in inference mode on custom dataset"
            )

        if lbl_set_codes is None:
            lbl_set_codes = [p.strip() for p in args.inference_label_sets.split(",")]

        # When you finetune on a custom dataset, you essentially replace the
        # original train/val/test splits (e.g., from the BindingMOAD) with the
        # new splits from the custom data. In some circumstances, you might want
        # to include additional fragments in the label set. You could specify
        # these fragments using the --inference_label_sets="custom_frags.smi".
        # However, for convenience, you can also simply use all fragments from
        # BindingMOAD, in addition to those from a .smi file.

        # If using a custom dataset, it's useful to generate a large fragment
        # library derived from the BindingMOAD dataset (all ligands), plus any
        # additional fragments that result from fragmenting the ligands in the
        # custom set (which may be in BindingMOAD, but may not be). If you use
        # --inference_label_sets="all", all these fragments wil be placed in a
        # single cache (.bin) file for quickly loading later.
        if "all" in lbl_set_codes:
            # Get the location of the every_csv file
            parent_every_csv = os.path.join(args.every_csv, os.pardir)
            parent_every_csv = os.path.relpath(parent_every_csv)

            # Get the locations of (possibly) cached label set files
            label_set_fps_bin = (
                parent_every_csv
                + os.sep
                + args.fragment_representation
                + "_all_label_set_fps.bin"
            )
            label_set_smis_bin = (
                parent_every_csv
                + os.sep
                + args.fragment_representation
                + "_all_label_set_smis.bin"
            )

            if os.path.exists(label_set_fps_bin) and os.path.exists(label_set_smis_bin):
                # Cache file exists, so load from that.
                with open(label_set_fps_bin, "rb") as file:
                    label_set_fps: torch.Tensor = pickle.load(file)
                    file.close()
                with open(label_set_smis_bin, "rb") as file:
                    label_set_smis: List[str] = pickle.load(file)
                    file.close()
            else:
                # Cache file does not exist, so generate.
                assert existing_label_set_entry_infos is not None, (
                    "Must provide existing label set entry infos when generating "
                    "a label set from scratch"
                )
                label_set_fps, label_set_smis = remove_redundant_fingerprints(
                    existing_label_set_fps,
                    existing_label_set_entry_infos,
                    device=device,
                )

                label_set_fps, label_set_smis = self._add_to_label_set(
                    args,
                    data_interface,
                    voxel_params,
                    device,
                    label_set_fps,
                    label_set_smis,
                    None,
                )

                # Save to cache file.
                with open(label_set_fps_bin, "wb") as file:
                    pickle.dump(label_set_fps, file)
                    file.close()
                with open(label_set_smis_bin, "wb") as file:
                    pickle.dump(label_set_smis, file)
                    file.close()

        # TODO: Cesar: label_set_fps and label_set_smis can be unbound. Good to check
        # with Cesar.

        # Add to that fingerprints from an SMI file.
        label_set_fps, label_set_smis = self._add_fingerprints_from_smis(
            args, lbl_set_codes, label_set_fps, label_set_smis, device
        )

        # self.model_parent.debug_smis_match_fps(label_set_fps, label_set_smis, device, args)

        print(f"Label set size: {len(label_set_fps)}")

        return label_set_fps, label_set_smis

    def _validate_run_test(self, args: Namespace, ckpt_filename: Optional[str]):
        """Validate the arguments required to run inference.

        Args:
            args (Namespace): The arguments.
            ckpt_filename (Optional[str]): The checkpoint.

        Raises:
            ValueError: If the arguments are invalid.
        """
        if not ckpt_filename:
            raise ValueError("Must specify a checkpoint in test mode")
        elif args.load_splits:
            # TODO: Cesar: See run_tests.sh. Why not?
            raise Exception(
                "To run the inference mode on a custom set is not required loading a previously saved test dataset"
            )
        elif not args.inference_label_sets:
            raise ValueError(
                "Must specify a label set (--inference_label_sets argument)"
            )
        elif not args.every_csv or not args.data_dir:
            raise Exception(
                "To run use a custom test set, you must specify the --every_csv and --data_dir arguments (for MOAD database)"
            )
        elif not args.custom_test_set_dir:
            raise Exception(
                "To run use a custom test set, you must specify the location of the dataset (--custom_test_set_dir argument) comprised of protein-ligand pairs (PDB file and SDF file)"
            )
        elif (
            "all" not in args.inference_label_sets
            or "train" in args.inference_label_sets
            or "test" in args.inference_label_sets
            or "val" in args.inference_label_sets
        ):
            raise Exception(
                "To run use a custom test set, you must only include the `all` label set or SMILES strings"
            )

    def _read_datasets_to_run_test(
        self, args: Namespace, voxel_params: VoxelParams
    ) -> Tuple[PdbSdfDirInterface, ParentInterface]:
        """Read the datasets required to run inference.

        Args:
            args (Namespace): The arguments.
            voxel_params (VoxelParams): The voxel parameters.

        Returns:
            Tuple[PdbSdfDirInterface, MOADInterface]: PdbSdfDirInterface contains the dataset to run test on,
                whereas MOADInterface contains the fragments to be used in the prediction.
        """
        print("Loading MOAD database.")
        moad = self._read_BindingMOAD_database(args, voxel_params)

        print("Loading custom database.")
        dataset = PdbSdfDirInterface(
            structures_dir=args.custom_test_set_dir,
            cache_pdbs_to_disk=args.cache_pdbs_to_disk,
            grid_width=voxel_params.width,
            grid_resolution=voxel_params.resolution,
            noh=args.noh,
            discard_distant_atoms=args.discard_distant_atoms,
        )

        return dataset, moad

    def _get_load_splits(self, args):
        """Get the splits to load. This is not required for inference."""
        return None

    def _get_cache(self, args):
        """Get the cache. This is not required for inference."""
        return None

    def _get_json_name(self, args):
        """Get the JSON name."""
        return "predictions_CustomDataset"

    def _save_examples_used(self, model, args):
        """Save the examples used. This is not required for inference."""
        pass
