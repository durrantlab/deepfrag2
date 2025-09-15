"""A model for inference."""

from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.types import StructureEntry, StructuresSplit
from collagen.model_parents.moad_voxel.test import VoxelModelTest
from argparse import Namespace
import os
from collagen.model_parents.moad_voxel.test_inference_utils import (
    remove_redundant_fingerprints,
)
import torch  # type: ignore
from typing import Any, List, Optional, Tuple
from collagen.core.voxelization.voxelizer import VoxelParams
import pickle


class Inference(VoxelModelTest):

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
        valid_in_house_smi_names = [
            "all_all", "all_test", "gte_4_acid_all", "gte_4_acid_test",
            "gte_4_aliphatic_all", "gte_4_aliphatic_test", "gte_4_aromatic_all",
            "gte_4_aromatic_test", "gte_4_base_all", "gte_4_base_test",
            "gte_4_all", "gte_4_test", "lte_3_all", "lte_3_test"
        ]

        for elem in args.inference_label_sets.split(','):
            is_disallowed_keyword = elem in ["train_on_moad", "train_on_complexes", "test_on_moad", "test_on_complexes", "val"]
            is_valid_format = (
                elem == "all" or
                elem.endswith(".smi") or
                elem.endswith(".smiles") or
                elem in valid_in_house_smi_names
            )
            if is_disallowed_keyword or not is_valid_format:
                raise Exception(
                    "Must specify the --inference_label_sets parameter with 'all', a path to a .smi/.smiles file, "
                    "a pre-compiled fragment set name, or a comma-separated combination of these."
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

        label_set_fps = None
        label_set_smis = []

        # If using a custom dataset, it's useful to generate a large fragment
        # library derived from the BindingMOAD dataset (all ligands), plus any
        # additional fragments that result from fragmenting the ligands in the
        # custom set (which may be in BindingMOAD, but may not be). If you use
        # --inference_label_sets="all", all these fragments wil be placed in a
        # single cache (.bin) file for quickly loading later.
        if "all" in lbl_set_codes:
            # Get the location of the csv file
            parent_csv = os.path.join(args.csv, os.pardir)
            parent_csv = os.path.relpath(parent_csv)

            # Get the locations of (possibly) cached label set files
            label_set_fps_bin = (
                parent_csv
                + os.sep
                + args.fragment_representation
                + "_all_label_set_fps.bin"
            )
            label_set_smis_bin = (
                parent_csv
                + os.sep
                + args.fragment_representation
                + "_all_label_set_smis.bin"
            )

            if os.path.exists(label_set_fps_bin) and os.path.exists(label_set_smis_bin):
                # Cache file exists, so load from that.
                with open(label_set_fps_bin, "rb") as file:
                    label_set_fps: torch.Tensor = torch.load(file, map_location=torch.device('cpu')).to(device)
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
                    torch.save(label_set_fps, file)
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
            raise ValueError(
                "Must specify the --ckpt_filename parameter to run the inference mode. This parameter contains the path "
                "to the DeepFrag model (.ckpt file) to be used."
            )
        elif not args.inference_label_sets:
            raise Exception(
                "Must specify the --inference_label_sets parameter. For example, 'gte_4_all', 'all', or a path to a SMILES file."
            )
        elif args.csv and args.data_dir and "all" not in args.inference_label_sets:
            raise Exception(
                "The --inference_label_sets parameter must contain the 'all' value when using the --csv and"
                " --data_dir parameters"
            )
        elif args.load_splits:
            raise Exception(
                "There are not training, validation, or test sets in inference mode"
            )
        
        valid_in_house_smi_names = [
            "all_all", "all_test", "gte_4_acid_all", "gte_4_acid_test",
            "gte_4_aliphatic_all", "gte_4_aliphatic_test", "gte_4_aromatic_all",
            "gte_4_aromatic_test", "gte_4_base_all", "gte_4_base_test",
            "gte_4_all", "gte_4_test", "lte_3_all", "lte_3_test"
        ]

        for elem in args.inference_label_sets.split(','):
            is_disallowed_keyword = elem in ["train_on_moad", "train_on_complexes", "test_on_moad", "test_on_complexes", "val"]
            is_valid_format = (
                elem == "all" or
                elem.endswith(".smi") or
                elem.endswith(".smiles") or
                elem in valid_in_house_smi_names
            )

            if is_disallowed_keyword or not is_valid_format:
                raise Exception(
                    "Must specify the --inference_label_sets parameter with 'all', a path to a .smi/.smiles file, "
                    "a pre-compiled fragment set name, or a comma-separated combination of these."
                )

    def _get_load_splits(self, args):
        """Get the splits to load. This is not required for inference."""
        return None

    def _get_cache(self, args):
        """Get the cache. This is not required for inference."""
        return None

    def _get_json_name(self, args):
        """Get the JSON name."""
        return "predictions_Multiple_Complexes"

    def _save_examples_used(self, model, args):
        """Save the examples used. This is not required for inference."""
        pass
