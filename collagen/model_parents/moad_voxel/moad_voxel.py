"""Parent class for all MOAD voxel models."""

from argparse import ArgumentParser, Namespace
import json
from typing import Type, TypeVar, List, Union
from collagen.model_parents.moad_voxel.inference import MoadVoxelModelInference
from collagen.model_parents.moad_voxel.inference_custom_dataset import MoadVoxelModelInferenceCustomSet
from collagen.model_parents.moad_voxel.inits import MoadVoxelModelInits
from collagen.model_parents.moad_voxel.test import MoadVoxelModelTest
from collagen.model_parents.moad_voxel.train import MoadVoxelModelTrain
from collagen.model_parents.moad_voxel import arguments
import os
from collagen.model_parents.moad_voxel.utils import MoadVoxelModelUtils
from collagen.core.molecules.fingerprints import download_molbert_ckpt
import pytorch_lightning as pl
import torch

from ... import VoxelParams

ENTRY_T = TypeVar("ENTRY_T")
TMP_T = TypeVar("TMP_T")
OUT_T = TypeVar("OUT_T")


class MoadVoxelModelParent(
    MoadVoxelModelInits,
    MoadVoxelModelTrain,
    MoadVoxelModelInference,
    MoadVoxelModelUtils,
):

    """Parent class for all MOAD voxel models."""

    # TODO: I should not have done this multiple-inheritance thing. Need to
    # refactor. Very confusing. 

    def __init__(
            self,
            model_cls: Type[pl.LightningModule],
            dataset_cls: Type[torch.utils.data.Dataset],
    ):
        """Initialize the model parent.
        
        Args:
            model_cls (Type[pl.LightningModule]): The model class. Soemthing 
                like DeepFragModelSDFData or DeepFragModel.
            dataset_cls (Type[torch.utils.data.Dataset]): The dataset class.
                Something like MOADFragmentDataset.
        """
        self.model_cls = model_cls
        self.dataset_cls = dataset_cls

        self.disable_warnings()

    @staticmethod
    def add_moad_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add the MOAD arguments to the parser.
        
        Args:
            parent_parser (ArgumentParser): The parser to add the arguments to.
            
        Returns:
            ArgumentParser: The parser with the arguments added.
        """
        return arguments.add_moad_args(parent_parser)

    @staticmethod
    def fix_moad_args(args: Namespace) -> Namespace:
        """Only works after arguments have been parsed, so in a separate
        definition.
        
        Args:
            args (Namespace): The arguments parsed by argparse.

        Returns:
            Namespace: The arguments with the MOAD arguments fixed.
        """
        return arguments.fix_moad_args(args)

    @staticmethod
    def pre_voxelize(
            args: Namespace, voxel_params: VoxelParams, entry: ENTRY_T
    ) -> TMP_T:
        """Preprocess the entry before voxelization. Should be overwritten by
        child class.
        
        Args:
            args (Namespace): The arguments parsed by argparse.
            voxel_params (VoxelParams): The voxelization parameters.
            entry (ENTRY_T): The entry to preprocess.
            
        Returns:
            TMP_T: The preprocessed entry.
        """
        return entry

    @staticmethod
    def voxelize(
            args: Namespace,
            voxel_params: VoxelParams,
            device: torch.device,
            batch: List[TMP_T],
    ) -> OUT_T:
        """Voxelize the batch. Should be overwritten by child class.
        
        Args:
            args (Namespace): The arguments parsed by argparse.
            voxel_params (VoxelParams): The voxelization parameters.
            device (torch.device): The device to use.
            batch (List[TMP_T]): The batch to voxelize.
            
        Returns:
            OUT_T: The voxelized batch.
        """
        raise NotImplementedError()

    @staticmethod
    def batch_eval(args: Namespace, batch: OUT_T):
        """Evaluate the batch. Should be overwritten by child class.

        Args:
            args (Namespace): The arguments parsed by argparse.
            batch (OUT_T): The batch to evaluate.
        """
        pass

    @staticmethod
    def custom_test(args: Namespace, predictions):
        """Run custom test. Should be overwritten by child class.

        Args:
            args (Namespace): The arguments parsed by argparse.
            predictions: The predictions.
        """
        pass

    @staticmethod
    def setup_fingerprint_scheme(args: Namespace):
        """Set up the fingerprint scheme.
        
        Args:
            args (Namespace): The arguments parsed by argparse.
        """
        if args.fragment_representation in ["rdk10", "rdk10_x_morgan", "random_2048"]:
            args.__setattr__("fp_size", 2048)
        elif args.fragment_representation in [
            "molbert",
            "binary_molbert",
            "normalized_molbert",
            "random_1536",
        ]:
            args.__setattr__("fp_size", 1536)
            download_molbert_ckpt()
        else:
            raise Exception("The fragment representation is wrong.")

    def load_checkpoint(self, args: Namespace = None, validate_args=True) -> Union[str, None]:
        """Load the checkpoint.

        Args:
            args (Namespace): The arguments parsed by argparse.
            validate_args (bool): Whether to validate the arguments.

        Returns:
            Union[str, None]: The checkpoint filename to load.
        """
        ckpt_filename = self.get_checkpoint_filename(args, validate_args)
        if ckpt_filename is not None:
            print(f"Restoring from checkpoint: {ckpt_filename}")

        return ckpt_filename

    def run(self, args: Namespace = None):
        """Run he model.

        Args:
            args (Namespace): The arguments parsed by argparse.
        """
        self.disable_warnings()
        self.setup_fingerprint_scheme(args)
        ckpt_filename = self.load_checkpoint(args)

        if args.mode == "train":
            print("Starting 'training' process")
            self.run_train(args, ckpt_filename)
        elif args.mode == "warm_starting":
            print("Starting 'warm_starting' process")
            self.run_warm_starting(args)
        elif args.mode == "test":
            print("Starting 'test' process")
            MoadVoxelModelTest(self).run_test(args, ckpt_filename)
        elif args.mode == "inference":
            print("Starting 'inference' process")
            self.run_inference(args, ckpt_filename)
        elif args.mode == "inference_custom_set":
            print("Starting 'inference_custom_set' process")
            MoadVoxelModelInferenceCustomSet(self).run_test(args, ckpt_filename)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    # def _get_train_frag_counts(self, args, moad, train, voxel_params, device):
    #     # Without calculating voxels and fingerprints...

    #     # NOTE: No longer used, but leaving here as an example of how to get
    #     # fragments without calculating voxels. Useful for testing?

    #     voxel_params_frag = copy.deepcopy(voxel_params)
    #     voxel_params_frag.calc_voxels = False
    #     voxel_params_frag.calc_fps = False
    #     train_data_to_get_frags = get_data_from_split(self,
    #         args, moad, train, voxel_params_frag, device
    #     )

    #     frag_counts = {}

    #     for payload in tqdm(train_data_to_get_frags, desc="Counting fragment SMILES..."):
    #         _, _, entry_infos = payload
    #         for entry_info in entry_infos:
    #             if entry_info.fragment_smiles not in frag_counts:
    #                 # Don't use set here. If one ligand has multiple identical
    #                 # fragments, I want them all listed.
    #                 frag_counts[entry_info.fragment_smiles] = 0
    #             frag_counts[entry_info.fragment_smiles] += 1
    #     return frag_counts

    def save_examples_used(self, model: pl.LightningModule, args: Namespace):
        """Save the examples used.
        
        Args:
            model (pl.LightningModule): The model.
            args (Namespace): The arguments parsed by argparse.
        """
        if args.default_root_dir is None:
            pth = os.getcwd() + os.sep
        else:
            pth = args.default_root_dir + os.sep

        if args.mode == "train":
            torch.save(model.state_dict(), f"{pth}model_train.pt")
        elif args.mode == "warm_starting":
            torch.save(model.state_dict(), f"{pth}model_fine_tuned.pt")

        out_name = pth + os.sep + args.mode + ".actually_used.json"
        if not os.path.exists(out_name):
            examples_used = model.get_examples_actually_used()
            with open(out_name, "w") as f:
                json.dump(examples_used, f, indent=4)
