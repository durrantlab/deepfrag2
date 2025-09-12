"""Utilities for MOAD voxel model."""

from argparse import Namespace
from typing import Optional, TYPE_CHECKING, List
from collagen.checkpoints import get_last_checkpoint
from collagen.core.loader import DataLambda, MultiLoader
from collagen.core.voxelization.voxelizer import VoxelParams
from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.types import StructuresSplit
import torch  # type: ignore
import os
import wget
import sys

if TYPE_CHECKING:
    from collagen.model_parents.moad_voxel.moad_voxel import VoxelModelParent


class VoxelModelUtils(object):
    """Provides utility funcitons."""

    def __init__(self, parent: "VoxelModelParent"):
        """Initialize the class.

        Args:
            parent (VoxelModelParent): The parent class.
        """
        self.parent = parent

    @staticmethod
    def disable_warnings():
        """Disable warnings from RDKit and ProDy."""
        from rdkit import RDLogger  # type: ignore
        import prody  # type: ignore

        RDLogger.DisableLog("rdApp.*")
        prody.confProDy(verbosity="none")

    def get_data_from_split(
        self,
        cache_file: str,
        args: Namespace,
        data_interface: ParentInterface,
        voxel_params: VoxelParams,
        device: torch.device,
        # NOTE: `split` needs to be optional because no split when running
        # inference. See inference_multiple_complexes.py.
        split: Optional[StructuresSplit],
        shuffle=True,
    ) -> DataLambda:
        """Where you do actual dataset construction. The transform
        function actually gets the data (voxelizes and creates fingerprint).
        Note also call to self.dataset_cls should create the .json.cache file.
        TODO: Create separate function .pre_voxelize_with_voxel that just
        calculates just fingerprint.

        Args:
            self: This object
            cache_file (str): The cache file to use.
            args (Namespace): The user arguments.
            data_interface (ParentInterface): The dataset interface.
            voxel_params (VoxelParams): Parameters for voxelization.
            device (torch.device): The device to use.
            split (Optional[StructuresSplit]): The split to use.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Returns:
            DataLambda: The data.
        """
        # JDD NOTE: self.dataset_cls could be something like MOADFragmentDataset

        dataset = self.parent.dataset_cls(
            data_interface=data_interface,
            cache_file=cache_file,
            cache_cores=args.num_dataloader_workers,
            split=split,
            transform=(
                lambda entry: self.parent.pre_voxelize(
                    args, entry
                )
            ),
            args=args,
        )

        return (
            MultiLoader(
                dataset,
                args.fragment_representation,
                shuffle=shuffle,
                num_dataloader_workers=args.num_dataloader_workers,
                max_voxels_in_memory=args.max_voxels_in_memory,
            ).batch(args.batch_size)
            # self.__class__.voxelize below actually makes the voxel and
            # fragments, I think. For fragments only, use
            # voxel_params.frag_fpts_only = True
            .map(
                lambda batch: self.parent.voxelize(
                    args, device, batch
                )
            )
        )

    def resolve_and_download_smi_files(
        self, smi_codes: List[str]
    ) -> List[str]:
        """Resolve SMILES file short names to local paths, downloading if needed.

        Args:
            smi_codes (List[str]): A list of SMILES file short names or paths.

        Returns:
            List[str]: A list of local paths to the SMILES files.
        """
        url_by_in_house_smi = {
            "all_all": "https://durrantlab.pitt.edu/apps/deepfrag2/models/all_all_sets_frags.smi",
            "all_test": "https://durrantlab.pitt.edu/apps/deepfrag2/models/all_test_set_frags.smi",
            "gte_4_acid_all": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_acid_all_sets_frags.smi",
            "gte_4_acid_test": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_acid_test_set_frags.smi",
            "gte_4_aliphatic_all": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_aliphatic_all_sets_frags.smi",
            "gte_4_aliphatic_test": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_aliphatic_test_set_frags.smi",
            "gte_4_aromatic_all": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_aromatic_all_sets_frags.smi",
            "gte_4_aromatic_test": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_aromatic_test_set_frags.smi",
            "gte_4_base_all": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_base_all_sets_frags.smi",
            "gte_4_base_test": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_base_test_set_frags.smi",
            "gte_4_all": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_all_sets_frags.smi",
            "gte_4_test": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_test_set_frags.smi",
            "lte_3_all": "https://durrantlab.pitt.edu/apps/deepfrag2/models/lte_3_all_sets_frags.smi",
            "lte_3_test": "https://durrantlab.pitt.edu/apps/deepfrag2/models/lte_3_test_set_frags.smi",
        }
        resolved_paths = []
        for code in smi_codes:
            if code in url_by_in_house_smi:
                filename = os.path.basename(url_by_in_house_smi[code])
                resolved_paths.append(
                    VoxelModelUtils.__download_deepfrag_smi(
                        filename, url_by_in_house_smi[code]
                    )
                )
            elif code.endswith(".smi") or code.endswith(".smiles"):
                resolved_paths.append(code)
        return resolved_paths

    @staticmethod
    def get_checkpoint_filename(
        args: Namespace, validate_args: bool = True
    ) -> Optional[str]:
        """Get the checkpoint filename.

        Args:
            args (Namespace): The arguments parsed by argparse.
            validate_args (bool, optional): Whether to validate the arguments.
                Defaults to True.

        Raises:
            ValueError: If the arguments are invalid.

        Returns:
            Optional[str]: The checkpoint filename.
        """
        url_by_in_house_model = {
            "all_best": "https://durrantlab.pitt.edu/apps/deepfrag2/models/all_best.ckpt",
            "gte_4_acid_best": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_acid_best.ckpt",
            "gte_4_aliphatic_best": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_aliphatic_best.ckpt",
            "gte_4_aromatic_best": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_aromatic_best.ckpt",
            "gte_4_base_best": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_base_best.ckpt",
            "gte_4_best": "https://durrantlab.pitt.edu/apps/deepfrag2/models/gte_4_best.ckpt",
            "lte_3_best": "https://durrantlab.pitt.edu/apps/deepfrag2/models/lte_3_best.ckpt",
        }

        if validate_args:
            if args.load_checkpoint and args.load_newest_checkpoint:
                raise ValueError(
                    "Can specify 'load_checkpoint=xyz' or 'load_newest_checkpoint' but not both."
                )
            if args.model_for_warm_starting and (args.load_checkpoint or args.load_newest_checkpoint):
                raise ValueError(
                    "If warm starting will be performed, then it cannot specify 'load_checkpoint=xyz' nor 'load_newest_checkpoint'."
                )
            if args.mode == "warm_starting" and not args.model_for_warm_starting:
                raise ValueError(
                    "If 'warm_starting' mode was specified, then it must be specified the 'model_for_warm_starting' parameter."
                )
            if args.model_for_warm_starting and args.mode != "warm_starting":
                raise ValueError(
                    "The 'model_for_warm_starting' parameter is only valid when 'warm_starting' mode is specified."
                )
            if args.load_checkpoint and args.load_checkpoint in url_by_in_house_model and not ("inference_" in args.mode):
                raise ValueError(
                    "The in-house models can be only used in inference mode."
                )

        ckpt = None
        if args.load_checkpoint:
            if args.load_checkpoint in url_by_in_house_model:
                args.load_checkpoint = VoxelModelUtils.__download_deepfrag_ckpt(
                    args.load_checkpoint + ".ckpt",
                    url_by_in_house_model[args.load_checkpoint])

            ckpt = args.load_checkpoint
        elif args.load_newest_checkpoint:
            ckpt = get_last_checkpoint(args)

        return ckpt

    @staticmethod
    def __download_deepfrag_ckpt(deepfrag_model_ckpt, deepfrag_model_url):
        """Download an in-house DeepFrag model checkpoint."""

        current_directory = os.getcwd() + os.sep + "pretrained_models"
        if not os.path.exists(current_directory):
            os.makedirs(current_directory, exist_ok=True)

        deepfrag_model_path = current_directory + os.sep + deepfrag_model_ckpt
        if not os.path.exists(deepfrag_model_path):
            print(f"Downloading DeepFrag2 model {deepfrag_model_ckpt} from {deepfrag_model_url}")
            try:
                wget.download(
                    deepfrag_model_url,
                    deepfrag_model_path,
                    VoxelModelUtils.__bar_progress,
                )
            except Exception as e:
                print("")
                assert False, f"Unable to download file {deepfrag_model_url} to {deepfrag_model_path}. Please place the file manually in the pretrained_models directory and try again."
        return deepfrag_model_path

    @staticmethod
    def __download_deepfrag_smi(smi_filename, smi_url):
        """Download an in-house DeepFrag SMILES file."""
        current_directory = os.getcwd() + os.sep + "pretrained_models"
        if not os.path.exists(current_directory):
            os.makedirs(current_directory, exist_ok=True)
        smi_path = current_directory + os.sep + smi_filename
        if not os.path.exists(smi_path):
            print("Starting download of the DeepFrag SMILES file: ", smi_filename)
            wget.download(
                smi_url,
                smi_path,
                VoxelModelUtils.__bar_progress_smi,
            )
        return smi_path

    @staticmethod
    def __bar_progress(current: float, total: float, width=80):
        """Progress bar for downloading Molbert model.

        Args:
            current (float): Current progress.
            total (float): Total progress.
            width (int, optional): Width of the progress bar. Defaults to 80.
        """
        progress_message = "Downloading DeepFrag model: %d%% [%d / %d] bytes" % (
            current / total * 100,
            current,
            total,
        )
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    @staticmethod
    def __bar_progress_smi(current: float, total: float, width=80):
        """Progress bar for downloading a DeepFrag SMILES file.
        Args:
            current (float): Current progress.
            total (float): Total progress.
            width (int, optional): Width of the progress bar. Defaults to 80.
        """
        progress_message = "Downloading DeepFrag SMILES file: %d%% [%d / %d] bytes" % (
            current / total * 100,
            current,
            total,
        )
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()