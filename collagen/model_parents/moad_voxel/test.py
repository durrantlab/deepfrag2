"""Test a model on the MOAD dataset."""

import glob
from argparse import Namespace
import cProfile
from io import StringIO
import json
import pstats
import os
import re
from collagen.core.loader import DataLambda
from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.types import StructureEntry, StructuresSplit
from collagen.external.moad.interface import MOADInterface
from collagen.external.paired_csv.interface import PairedCsvInterface
from collagen.external.pdb_sdf_dir.interface import PdbSdfDirInterface
from collagen.model_parents.moad_voxel.test_inference_utils import (
    remove_redundant_fingerprints,
)
import torch  # type: ignore
from tqdm.std import tqdm
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pickle
# import multiprocessing
# from torch import multiprocessing
from collagen.core.molecules.mol import mols_from_smi_file
from collagen.core.voxelization.voxelizer import VoxelParams
from collagen.metrics.ensembled import averaged as ensemble_helper
from collagen.external.common.split import create_train_val_test_splits
from collagen.metrics.metrics import (
    most_similar_matches,
    pca_space_from_label_set_fingerprints,
    top_k,
)
from collagen.core.molecules.smiles_utils import standardize_smiles_or_rdmol

if TYPE_CHECKING:
    import pytorch_lightning as pl  # type: ignore
    from collagen.metrics.metrics import PCAProject
    from collagen.model_parents.moad_voxel.moad_voxel import VoxelModelParent

# See https://github.com/pytorch/pytorch/issues/3492
# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass
# multiprocessing_ctx = multiprocessing.get_context("spawn")


def _return_paramter(object):
    """Return a parameter. For use in imap_unordered.

    Args:
        object (Any): The parameter.

    Returns:
        Any: The parameter returned.
    """
    return object


class VoxelModelTest(object):

    """A model for testing."""

    def __init__(self, parent: "VoxelModelParent"):
        """Initialize the VoxelModelTest object.

        Args:
            parent (VoxelModelParent): The model parent.
        """
        self.parent = parent

    def _add_to_label_set(
        self,
        args: Namespace,
        data_interface: ParentInterface,
        voxel_params: VoxelParams,
        device: torch.device,
        existing_label_set_fps: torch.Tensor,
        existing_label_set_smis: List[str],
        # NOTE: `split` needs to be optional because no split when running
        # inference. See inference_multiple_complexes.py.
        split: Optional[StructuresSplit] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Add fingerprints to a label set (lookup table). This function allows
        you to include additional fingerprints (e.g., from the training and
        validation sets) in the label set for testing. Takes the fingerprints
        right from the split. A helper script.

        Args:
            self: This object
            args (Namespace): The user arguments.
            data_interface (ParentInterface): The dataset interface.
            voxel_params (VoxelParams): Parameters for voxelization.
            device (torch.device): The device to use.
            existing_label_set_fps (torch.Tensor): The existing tensor of
                fingerprints to which these new ones should be added.
            existing_label_set_smis (List[str]): The existing list of SMILES
                strings to which the new ones should be added.
            split (Optional[StructuresSplit]): The splits of the MOAD dataset.

        Returns:
            Tuple[torch.Tensor, List[Union[str, StructureEntry]]]: The updated
                fingerprint tensor and smiles list.
        """
        # global multiprocessing_ctx

        # TODO: Harrison: How hard would it be to make it so data below doesn't
        # voxelize the receptor? Is that adding a lot of time to the
        # calculation? Just a thought. See other TODO: note about this.
        data = self.parent.utils.get_data_from_split(
            cache_file=args.cache,
            args=args,
            data_interface=data_interface,
            split=split,
            voxel_params=voxel_params,
            device=device,
            shuffle=False,
        )

        all_fps = []
        all_smis = []

        # TODO: When using multiprocessing below, causes CUDA errors on some
        # systems. But not on the CRC, so it's not a Windows vs. Linux issue.
        # Below is commented out to avoid error, but runs much slower.
        # Fortunately, this calculation is only performed once because the
        # result is cached.

        # with multiprocessing.Pool() as p:
        #     for batch in tqdm(
        #         p.imap_unordered(_return_paramter, data),
        #         total=len(data),
        #         desc=f"Getting fingerprints from {split.name if split else 'Full'} set..."
        #     ):
        #         voxels, fps_tnsr, smis = batch
        #         all_fps.append(fps_tnsr)
        #         all_smis.extend(smis)

        # Single-processor code to avoid error above.
        for batch in tqdm(
            data,
            desc=f"Getting fingerprints from {split.name if split else 'Full'} set...",
        ):
            voxels, fps_tnsr, smis = batch
            all_fps.append(fps_tnsr)
            all_smis.extend(smis)

        if existing_label_set_smis is not None:
            all_smis.extend(existing_label_set_smis)
        if existing_label_set_fps is not None:
            all_fps.append(existing_label_set_fps)

        # Remove redundancies.
        fps_tnsr, all_smis = remove_redundant_fingerprints(
            torch.cat(all_fps), all_smis, device
        )

        return fps_tnsr, all_smis

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
        """Create a label set (look-up) tensor and smiles list for testing.
        Can be comprised of the fingerprints in the train and/or test and/or
        val sets, as well as SMILES strings from a file.

        Args:
            self: This object
            args (Namespace): The user arguments.
            device (torch.device): The device to use.
            data_interface (ParentInterface): The data interface. Defaults to
                None.
            voxel_params (VoxelParams): Parameters for voxelization. Defaults
                to None.
            existing_label_set_fps (torch.Tensor, optional): The existing
                tensor of fingerprints to which these new ones should be added.
                Defaults to None.
            existing_label_set_entry_infos (List[StructureEntry], optional):
                _description_. Defaults to None.
            skip_test_set (bool, optional): Do not add test-set fingerprints,
                presumably because they are already present in
                existing_label_set_entry_infos. Defaults to False.
            train (StructuresSplit, optional): The train split. Defaults to
                None.
            val (StructuresSplit, optional): The val split. Defaults to None.
            test (StructuresSplit, optional): The test split. Defaults to None.
            lbl_set_codes (List[str], optional): _description_. Defaults to
                None.

        Returns:
            Tuple[torch.Tensor, List[Union[str, StructureEntry]]]: The updated
                fingerprint tensor and smiles list.
        """
        if "all" in args.inference_label_sets:  # type: ignore
            raise Exception(
                "The 'all' value for the --inference_label_sets parameter is not a valid value in test mode"
            )

        # skip_test_set can be true if those fingerprints are already in
        # existing_label_set

        if lbl_set_codes is None:
            lbl_set_codes = [p.strip() for p in args.inference_label_sets.split(",")]  # type: ignore

        # Load from train, val, and test sets.
        if existing_label_set_fps is None:
            label_set_fps = torch.zeros(
                (0, args.fp_size),  # type: ignore
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            )
            label_set_smis = []
        else:
            # If you get an existing set of fingerprints, be sure to keep only the
            # unique ones.
            assert (
                existing_label_set_entry_infos is not None
            ), "Must provide entry infos with existing fingerprints"
            label_set_fps, label_set_smis = remove_redundant_fingerprints(
                existing_label_set_fps, existing_label_set_entry_infos, device=device
            )

        if (
            "train" in lbl_set_codes
            and train_split is not None
            and len(train_split.targets) > 0
        ):
            label_set_fps, label_set_smis = self._add_to_label_set(
                args,
                data_interface,
                voxel_params,
                device,
                label_set_fps,
                label_set_smis,
                train_split,
            )

        if (
            "val" in lbl_set_codes
            and val_split is not None
            and len(val_split.targets) > 0
        ):
            label_set_fps, label_set_smis = self._add_to_label_set(
                args,
                data_interface,
                voxel_params,
                device,
                label_set_fps,
                label_set_smis,
                val_split,
            )

        if (
            "test" in lbl_set_codes
            and not skip_test_set
            and test_split is not None
            and len(test_split.targets) > 0
        ):
            label_set_fps, label_set_smis = self._add_to_label_set(
                args,
                data_interface,
                voxel_params,
                device,
                label_set_fps,
                label_set_smis,
                test_split,
            )

        # Add to that fingerprints from an SMI file.
        label_set_fps, label_set_smis = self._add_fingerprints_from_smis(
            args, lbl_set_codes, label_set_fps, label_set_smis, device
        )

        # self.parent.debug_smis_match_fps(label_set_fps, label_set_smis, device, args)

        print(f"Label set size: {len(label_set_fps)}")

        return label_set_fps, label_set_smis

    def _add_fingerprints_from_smis(
        self,
        args: Namespace,
        lbl_set_codes: List[str],
        label_set_fps: Optional[torch.Tensor],
        label_set_smis: Optional[List[str]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Add fingerprints from an SMI file to the label set.

        Args:
            args (Namespace): The user arguments.
            lbl_set_codes (List[str]): The label set codes. train, val, test,
                all.
            label_set_fps (torch.Tensor): The label set fingerprints.
            label_set_smis (List[str]): The label set SMILES strings.
            device (torch.device): The device to use.

        Returns:
            Tuple[torch.Tensor, List[str]]: The updated label set fingerprints
                and SMILES strings.
        """
        if label_set_smis is None:
            label_set_smis = []
        smi_files_or_codes = [
            f for f in lbl_set_codes if f not in ["train", "val", "test", "all"]
        ]
        smi_files = self.parent.utils.resolve_and_download_smi_files(smi_files_or_codes)
        if smi_files:
            fp_tnsrs_from_smi_file = [label_set_fps] if len(label_set_smis) > 0 else []
            for filename in smi_files:
                filename_fps = filename + "_" + args.fragment_representation + "_fps.bin"
                if os.path.exists(filename_fps):
                    with open(filename_fps, "rb") as file:
                        dictionary_smi_fps: Dict[str, torch.Tensor] = torch.load(file, map_location=torch.device('cpu'))
                        for key in dictionary_smi_fps:
                            dictionary_smi_fps[key] = dictionary_smi_fps[key].to(device)
                        filename_fps = None
                        file.close()
                else:
                    dictionary_smi_fps = {}

                for smi, mol in mols_from_smi_file(filename):
                    if dictionary_smi_fps and smi in dictionary_smi_fps:
                        fp_tnsrs_from_smi_file.append(dictionary_smi_fps[smi])
                    else:
                        fps = torch.tensor(
                            mol.fingerprint(args.fragment_representation, args.fp_size),
                            dtype=torch.float32,
                            device=device,
                            requires_grad=False,
                        ).reshape((1, args.fp_size))
                        fp_tnsrs_from_smi_file.append(fps)
                        dictionary_smi_fps[smi] = fps

                    label_set_smis.append(StructureEntry(
                        fragment_smiles=smi,
                        parent_smiles=None,
                        receptor_name=None,
                        connection_pt=None,
                        ligand_id=None,
                        fragment_idx=None,
                    ))

                if filename_fps is not None:
                    with open(filename_fps, "wb") as file:
                        torch.save(dictionary_smi_fps, file)
                        file.close()
            label_set_fps = torch.cat(fp_tnsrs_from_smi_file)

            # Remove redundancy
            label_set_fps, label_set_smis = remove_redundant_fingerprints(
                label_set_fps, label_set_smis, device
            )

        return label_set_fps, label_set_smis

    def _on_first_checkpoint(
        self,
        all_test_data: Any,
        args: Namespace,
        model: "pl.LightningModule",
        train: StructuresSplit,
        val: StructuresSplit,
        data_interface: ParentInterface,
        lbl_set_codes: Optional[List[str]],
        device: torch.device,
        predictions_per_rot: ensemble_helper.AveragedEnsembled,
    ) -> Tuple["PCAProject", torch.Tensor, torch.Tensor, List[StructureEntry]]:
        """Certain variables can only be defined when processing the first
        checkpoint. Moving this out of the loop so it's not distracting. Note
        that all_test_data is modified in place and so does not need to be
        returned.

        Args:
            all_test_data (Any): The test data.
            args (Namespace): The user arguments.
            model (pl.LightningModule): The model.
            train (StructuresSplit): The training data.
            val (StructuresSplit): The validation data.
            data_interface (ParentInterface): The data interface.
            lbl_set_codes (List[str]): The label set codes.
            device (torch.device): The device to use.
            predictions_per_rot (ensemble_helper.AveragedEnsembled): The
                averaged ensemble predictions.

        Returns:
            Tuple[PCAProject, torch.Tensor, torch.Tensor, List[StructureEntry]]:
                The PCA space, the predictions (all zeros), the label set
                fingerprints, and the label set entry infos.
        """
        voxel_params = self.parent.voxel_params

        # Get the label set to use. Note that it only does this once (for the
        # first-checkpoint model), but I need model so I'm leaving it in the
        # loop.
        label_set_fingerprints, label_set_entry_infos = self._create_label_set(
            args,
            device,
            data_interface,
            voxel_params,
            existing_label_set_fps=predictions_per_rot.model.prediction_targets,
            existing_label_set_entry_infos=predictions_per_rot.model.prediction_targets_entry_infos,
            skip_test_set=True,
            train_split=train,
            val_split=val,
            lbl_set_codes=lbl_set_codes,
        )

        # Get a PCA (or other) space defined by the label-set fingerprints.
        pca_space = pca_space_from_label_set_fingerprints(label_set_fingerprints, 2)

        all_test_data["pcaPercentVarExplainedByEachComponent"] = [
            100 * r for r in pca_space.pca.explained_variance_ratio_.tolist()
        ]

        # Must be a list of StructureEntry
        assert isinstance(label_set_entry_infos, list) and isinstance(
            label_set_entry_infos[0], StructureEntry
        ), "label_set_entry_infos must be a list of StructureEntry"

        return (
            pca_space,
            torch.zeros(
                model.prediction_targets.shape,
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            ),
            torch.tensor(
                label_set_fingerprints,
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            ),
            label_set_entry_infos,  # type: ignore
        )

    def _run_test_on_single_checkpoint(
        self,
        all_test_data: Any,
        args: Namespace,
        model: "pl.LightningModule",
        ckpt_idx: int,
        ckpt: str,
        trainer: Any,
        test_data: DataLambda,
        train: StructuresSplit,
        val: StructuresSplit,
        data_interface: ParentInterface,
        lbl_set_codes: Optional[List[str]],
        avg_over_ckpts_of_avgs: Any,
        device: torch.device,
        pca_space: Optional["PCAProject"] = None,
        label_set_fingerprints: Optional[torch.Tensor] = None,
        label_set_entry_infos: Optional[List[StructureEntry]] = None,
    ) -> Union[
        Tuple["PCAProject", torch.Tensor, torch.Tensor, List[StructureEntry]], None
    ]:
        """Test run on a single checkpoint. You're iterating through
        multiple checkpoints. This allows output from multiple trained models to
        be averaged.

        Args:
            all_test_data (Any): The test data.
            args (Namespace): The user arguments.
            model (pl.LightningModule): The model.
            ckpt_idx (int): The checkpoint index.
            ckpt (str): The checkpoint.
            trainer (Any): The trainer.
            test_data (DataLambda): The test data.
            train (StructuresSplit): The training data.
            val (StructuresSplit): The validation data.
            data_interface (ParentInterface): The data interface.
            lbl_set_codes (List[str]): The label set codes.
            avg_over_ckpts_of_avgs (Any): The averaged ensemble predictions.
            device (torch.device): The device to use.
            pca_space (Optional[PCAProject], optional): The PCA space. Defaults
                to None (None on first checkpoint).
            label_set_fingerprints (Optional[torch.Tensor], optional): The label
                set fingerprints. Defaults to None (None on first checkpoint).
            label_set_entry_infos (Optional[List[StructureEntry]], optional): The
                label set entry infos. Defaults to None (None on first checkpoint).

        Returns:
            Union[Tuple["PCAProject", torch.Tensor, torch.Tensor, List[StructureEntry]], None]:
                The PCA space, the predictions (averaged over multiple
                checkpoints), the label set fingerprints, and the label set
                entry infos.
        """
        # all_test_data is modified in place and so does not need to be
        # returned.

        # Could pass these as parameters, but let's keep things simple and just
        # reinitialize.

        predictions_per_rot = ensemble_helper.AveragedEnsembled(
            trainer,
            model,
            test_data,
            args.rotations,
            device,
            ckpt,
            args.aggregation_rotations,
            args.fragment_representation,
            args.save_fps,
        )

        if ckpt_idx == 0:
            # Get the label set to use. Note that it only does this once (for
            # the first-checkpoint model), but I need model so I'm leaving it in
            # the loop. Other variables also defined here.
            (
                pca_space,
                avg_over_ckpts_of_avgs,
                label_set_fingerprints,
                label_set_entry_infos,
            ) = self._on_first_checkpoint(
                all_test_data,
                args,
                model,
                train,
                val,
                data_interface,
                lbl_set_codes,
                device,
                predictions_per_rot,
            )

        # This assertion should always be satisfied. pca_space only none if
        # ckpt_idx was 0, but in that case it is defined above.
        assert pca_space is not None, "PCA space must be defined"
        assert (
            label_set_fingerprints is not None
        ), "Label set fingerprints must be defined"
        assert (
            label_set_entry_infos is not None
        ), "Label set entry infos must be defined"

        predictions_per_rot.finish(pca_space)
        model, predictions_ensembled = predictions_per_rot.unpack()
        torch.add(
            avg_over_ckpts_of_avgs,
            predictions_ensembled,
            out=avg_over_ckpts_of_avgs,
        )

        if ckpt_idx == 0:
            for i in range(predictions_per_rot.predictions_ensembled.shape[0]):
                # Add in correct answers for all entries
                correct_answer = predictions_per_rot.get_correct_answer_info(i)
                all_test_data["entries"].append(
                    {"groundTruth": correct_answer, "perCheckpoint": []}
                )

        # Calculate top_k metric for this checkpoint
        # import pdb; pdb.set_trace()
        top_k_results = top_k(
            predictions_ensembled,
            torch.tensor(
                model.prediction_targets,
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            ),
            label_set_fingerprints,
            k=[1, 8, 16, 32, 64],
        )
        all_test_data["checkpoints"][ckpt_idx]["topK"] = {
            f"testTop{k}": float(top_k_results[k]) for k in top_k_results
        }

        # Add info about the per-rotation predictions
        for entry_idx in range(len(predictions_ensembled)):
            entry = predictions_per_rot.get_predictions_info(entry_idx)
            all_test_data["entries"][entry_idx]["perCheckpoint"].append(entry)

        # Find most similar matches
        most_similar = most_similar_matches(
            predictions_ensembled,
            label_set_fingerprints,
            label_set_entry_infos,
            args.num_inference_predictions,
            pca_space,
        )
        for entry_idx in range(len(predictions_ensembled)):
            # Add closest compounds from label set.
            for predicted_entry_info, cos_similarity, pca in most_similar[entry_idx]:
                if pca is None:
                    pca = [None]

                all_test_data["entries"][entry_idx]["perCheckpoint"][-1][
                    "averagedPrediction"
                ]["closestFromLabelSet"].append(
                    {
                        "smiles": standardize_smiles_or_rdmol(predicted_entry_info.fragment_smiles),
                        "cosSimilarityWithAvgPrediction": cos_similarity,
                        "pcaProjection": pca[0],
                    }
                )

        if ckpt_idx == 0:
            return (
                pca_space,
                avg_over_ckpts_of_avgs,
                label_set_fingerprints,
                label_set_entry_infos,
            )

        return None

    def _save_test_results_to_json(
        self,
        all_test_data: dict,
        s: StringIO,
        args: Namespace,
        pth: Optional[str] = None,
    ):
        """Save the test results to a carefully formatted JSON file.

        Args:
            all_test_data (dict): The test data.
            s (StringIO): The string buffer to write to.
            args (Namespace): The user arguments.
            pth (str, optional): The path to save the JSON file to. Defaults to None.
        """
        jsn = json.dumps(all_test_data, indent=4)
        jsn = re.sub(r"([\-0-9\.]+?,)\n +?([\-0-9\.])", r"\1 \2", jsn, 0, re.MULTILINE)
        jsn = re.sub(
            r"\[\n +?([\-0-9\.]+?), ([\-0-9\.,]+?)\n +?\]",
            r"[\1, \2]",
            jsn,
            0,
            re.MULTILINE,
        )
        jsn = re.sub(r"\"Receptor ", '"', jsn, 0, re.MULTILINE)
        jsn = re.sub(r"\n +?\"dist", ' "dist', jsn, 0, re.MULTILINE)

        if pth is None:
            pth = os.getcwd()

        pth = (
            pth
            + os.sep
            + self._get_json_name(args)
            + os.sep
            + args.aggregation_rotations
            + os.sep
        )

        assert pth is not None, "Path must be specified"

        os.makedirs(pth, exist_ok=True)
        num = len(glob.glob(f"{pth}*.json", recursive=False))

        with open(f"{pth}test_results-{num + 1}.json", "w") as f:
            f.write(jsn)
        with open(f"{pth}cProfile-{num + 1}.txt", "w+") as f:
            f.write(s.getvalue())

        # txt = ""
        # for entry in all_test_data["entries"]:
        #     txt += "Correct\n"
        #     txt += "\t".join([str(e) for e in entry["groundTruth"]["pcaProjection"]]) + "\t" + entry["groundTruth"]["fragmentSmiles"] + "\n"
        #     txt += "averagedPrediction\n"
        #     txt += "\t".join([str(e) for e in entry["averagedPrediction"]["pcaProjection"]]) + "\n"
        #     txt += "closestFromLabelSet\n"
        #     for close in entry["averagedPrediction"]["closestFromLabelSet"]:
        #         txt += "\t".join([str(e) for e in close["pcaProjection"]]) + "\t" + close["smiles"] + "\n"
        #     txt += "predictionsPerRotation\n"
        #     for pred_per_rot in entry["predictionsPerRotation"]:
        #         txt += "\t".join([str(e) for e in pred_per_rot]) + "\n"
        #     txt = txt + "\n"

        # open("/mnt/extra/test_results.txt", "w").write(txt)

        # pr.disable()
        # s = StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        # ps.print_stats()
        # with open('/mnt/extra/cProfile.txt', 'w+') as f:
        #    f.write(s.getvalue())

    def _validate_run_test(self, args: Namespace, ckpt_filename: Optional[str]):
        """Validate the arguments for the test mode.

        Args:
            args (Namespace): The user arguments.
            ckpt_filename (Optional[str]): The checkpoint to use.
        """
        if not ckpt_filename:
            raise ValueError("Must specify a checkpoint in test mode")
        elif not args.inference_label_sets:
            raise ValueError(
                "Must specify a label set (--inference_label_sets argument)"
            )
        elif args.csv and not args.data_dir:
            raise Exception(
                "To load the MOAD database, you must specify the --csv and --data_dir arguments"
            )
        elif not args.csv and not args.data_dir and not args.paired_data_csv:
            raise Exception(
                "To run the test mode, you must specify the --csv and --data_dir arguments for the MOAD database, the --data_dir argument for a non-paired database other than MOAD, or the --paired_data_csv argument for a paired database other than MOAD."
            )
        elif args.paired_data_csv and (args.csv or args.data_dir):
            raise Exception(
                "To run the test mode using a paired database other than MOAD database, you must only specify the --paired_data_csv argument."
            )
        elif args.paired_data_csv and args.data_dir:
            raise Exception(
                "For the test mode, you must only specify the --paired_data_csv argument to use a paired database other than MOAD database, or the --data_dir argument  to use a non-paired database other than MOAD database."
            )
        elif args.custom_test_set_dir:
            raise Exception(
                "To run the test mode must not be specified a custom dataset (--custom_test_set_dir argument)"
            )
        elif "all" in args.inference_label_sets:
            raise Exception(
                "The `all` value for label set (--inference_label_sets) must not be specified in test mode"
            )
        elif "test" not in args.inference_label_sets:
            raise ValueError(
                "To run in test mode, you must include the `test` label set"
            )
        elif not args.load_splits:
            raise Exception(
                "To run the test mode is required loading a previously saved test dataset"
            )
        elif args.load_splits and args.split_method != "random_default":
            raise Exception("You cannot specify --split_method if using --load_splits.")
        # TODO: jacob added below. Need to remember why added, make sure not redundant.
        elif args.test_predictions_file is not None and args.mode != "test":
            raise Exception(
                "You cannot specify --test_predictions_file unless you are in test mode."
            )

    def run_test(self, args: Namespace, ckpt_filename: str):
        """Run a model on the test and evaluates the output.

        Args:
            args (Namespace): The user arguments.
            ckpt_filename (str): The checkpoint to use.
        """
        pr = cProfile.Profile()
        pr.enable()

        self._validate_run_test(args, ckpt_filename)

        print(
            f"Using the operator {args.aggregation_rotations} to aggregate the inferences."
        )

        voxel_params = self.parent.voxel_params
        device = self.parent.inits.init_device(args)

        data_interface, set2run_test_on_single_checkpoint = self._read_datasets_to_run_test(
            args, voxel_params
        )

        train_split, val_split, test_split = create_train_val_test_splits(
            dataset=data_interface,
            seed=None,
            fraction_train=0.0,
            fraction_val=0.0,
            # Should be true to ensure independence when using
            # split_method="random". TODO: Could remove this parameter, force it
            # to be true. Also, could be user parameter.
            prevent_smiles_overlap=True,
            save_splits=None,
            load_splits=self._get_load_splits(args),
            max_pdbs_train=args.max_pdbs_train,
            max_pdbs_val=args.max_pdbs_val,
            max_pdbs_test=args.max_pdbs_test,
            split_method=args.split_method,  # NOTE: I don't think it matters what split is for testing.
            butina_cluster_cutoff=None,  # Hardcoded because no need to split test set.
        )

        # You'll always need the test data. Note that ligands are not fragmented
        # by calling the get_data_from_split function.
        test_data: "DataLambda" = self.parent.utils.get_data_from_split(
            cache_file=self._get_cache(args),
            args=args,
            data_interface=data_interface,
            split=test_split,
            voxel_params=voxel_params,
            device=device,
            shuffle=False,
        )
        print(f"Number of batches for the test data: {len(test_data)}")

        trainer = self.parent.inits.init_trainer(args)

        ckpts = [c.strip() for c in ckpt_filename.split(",")]
        all_test_data = {
            "checkpoints": [{"name": c, "order": i + 1} for i, c in enumerate(ckpts)],
            "entries": [],
        }

        model = None
        label_set_fingerprints = None
        pca_space = None
        label_set_entry_infos = None
        avg_over_ckpts_of_avgs = None
        for ckpt_idx, ckpt_filename in enumerate(ckpts):
            # You're iterating through multiple checkpoints. This allows output
            # from multiple trained models to be averaged.

            # Keep track of all fingerprints in case --test_predictions_file is
            # set. # TODO: I have the vague impression this is redundant with a
            # change cesar made, but not sure.
            all_fingerprints = []

            model = self.parent.inits.init_model(
                args, ckpt_filename, fragment_set=set2run_test_on_single_checkpoint
            )
            model.eval()
            payload = self._run_test_on_single_checkpoint(
                all_test_data,
                args,
                model,
                ckpt_idx,
                ckpt_filename,
                trainer,
                test_data,
                train_split,
                val_split,
                set2run_test_on_single_checkpoint,
                None,  # lbl_set_codes
                avg_over_ckpts_of_avgs,
                device,
                pca_space,
                label_set_fingerprints,
            )

            if ckpt_idx == 0:
                assert (
                    payload is not None
                ), "Payload should not be None if first checkpoint."

                # Payload is not None if first checkpoint.
                (
                    pca_space,
                    avg_over_ckpts_of_avgs,
                    label_set_fingerprints,
                    label_set_entry_infos,
                ) = payload

        assert model is not None, "Model should not be None after loop."
        assert (
            label_set_fingerprints is not None
        ), "Label set fingerprints should not be None after loop."
        assert pca_space is not None, "PCA space should not be None after loop."
        assert (
            label_set_entry_infos is not None
        ), "Label set entry infos should not be None after loop."
        assert (
            avg_over_ckpts_of_avgs is not None
        ), "Average over checkpoints of averages should not be None after loop."

        # Get the average of averages (across all checkpoints)
        torch.div(
            avg_over_ckpts_of_avgs,
            torch.tensor(len(ckpts), device=device),
            out=avg_over_ckpts_of_avgs,
        )

        # Calculate top-k metric of that average of averages
        top_k_results = top_k(
            avg_over_ckpts_of_avgs,
            torch.tensor(
                model.prediction_targets,
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            ),
            label_set_fingerprints,
            k=[1, 8, 16, 32, 64],
        )
        all_test_data["checkpoints"].append(
            {
                "name": "Using average fingerprint over all checkpoints",
                "topK": {f"testTop{k}": float(top_k_results[k]) for k in top_k_results},
            }
        )

        # Get the fingerprints of the average of average outputs.
        avg_over_ckpts_of_avgs_viz = pca_space.project(avg_over_ckpts_of_avgs)

        # For average of averages, find most similar matches
        most_similar = most_similar_matches(
            avg_over_ckpts_of_avgs,
            label_set_fingerprints,
            label_set_entry_infos,
            args.num_inference_predictions,
            pca_space,
        )
        for entry_idx in range(len(avg_over_ckpts_of_avgs)):
            # Add closest compounds from label set.
            all_test_data["entries"][entry_idx]["avgOfCheckpoints"] = {
                "pcaProjection": avg_over_ckpts_of_avgs_viz[entry_idx],
                "closestFromLabelSet": [],
            }
            for predicted_entry_info, cos_similarity, pca in most_similar[entry_idx]:
                if pca is None:
                    pca = [None]

                all_test_data["entries"][entry_idx]["avgOfCheckpoints"][
                    "closestFromLabelSet"
                ].append(
                    {
                        "smiles": predicted_entry_info.fragment_smiles,
                        "cosSimilarityWithAvgPrediction": cos_similarity,
                        "pcaProjection": pca[0],
                    }
                )

        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()

        self._save_test_results_to_json(all_test_data, s, args, args.default_root_dir)
        self._save_examples_used(model, args)

    def _read_datasets_to_run_test(
        self, args: Namespace, voxel_params: VoxelParams
    ) -> Tuple[ParentInterface, ParentInterface]:
        """Read the dataset to run test on.

        Args:
            args: The arguments passed to the program.
            voxel_params: The voxel parameters.

        Returns:
            A tuple of (ParentInterface, ParentInterface). The first
            coefficient contains the dataset to run test on, whereas the second
            coefficient contains the fragments to be used in the prediction. In
            test mode, both coefficients are the same ones, but in inference
            mode they are different.
        """
        # These two arguments can be used either to read the MOAD database or to read PDB/SDF files from a CSV file
        if args.csv and args.data_dir:
            # test mode on a Binding MOAD Database
            if args.mode == "test_on_moad":
                print("Loading MOAD database.")
                dataset = self._read_BindingMOAD_database(args, voxel_params)

            # test mode on PDB/SDF files
            elif args.mode == "test_on_complexes":
                print("Loading PDB/SDF files from a CSV file.")
                dataset = PdbSdfDirInterface(
                    metadata=args.csv,
                    structures_path=args.data_dir,
                    cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                    grid_width=voxel_params.width,
                    grid_resolution=voxel_params.resolution,
                    noh=args.noh,
                    discard_distant_atoms=args.discard_distant_atoms,
                )
        # test mode on a paired database
        elif args.paired_data_csv:
            dataset = PairedCsvInterface(
                structures=args.paired_data_csv,
                cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                grid_width=voxel_params.width,
                grid_resolution=voxel_params.resolution,
                noh=args.noh,
                discard_distant_atoms=args.discard_distant_atoms,
                use_prevalence=args.use_prevalence,
            )

        return dataset, dataset

    def _read_BindingMOAD_database(
        self, args: Namespace, voxel_params: VoxelParams
    ) -> MOADInterface:
        """Read the BindingMOAD database.

        Args:
            args: The arguments passed to the program.
            voxel_params: The voxel parameters.

        Returns:
            The MOAD database.
        """
        return MOADInterface(
            metadata=args.csv,
            structures_path=args.data_dir,
            cache_pdbs_to_disk=args.cache_pdbs_to_disk,
            grid_width=voxel_params.width,
            grid_resolution=voxel_params.resolution,
            noh=args.noh,
            discard_distant_atoms=args.discard_distant_atoms,
        )

    def _get_load_splits(self, args: Namespace) -> Any:
        """Get the splits to load.

        Args:
            args: The arguments passed to the program.

        Returns:
            The splits to load.
        """
        return args.load_splits

    def _get_cache(self, args: Namespace) -> Any:
        """Get the cache.

        Args:
            args: The arguments passed to the program.

        Returns:
            The cache.
        """
        return args.cache

    def _get_json_name(self, args: Namespace) -> str:
        """Get the name of the JSON file to save the results to.

        Args:
            args: The arguments passed to the program.

        Returns:
            The name of the JSON file to save the results to.
        """
        return (
            "predictions_MOAD" if "moad" in args.mode else "predictions_nonMOAD"
        )

    def _save_examples_used(self, model: "pl.LightningModule", args: Namespace):
        """Save the examples used.

        Args:
            model: The model.
            args: The arguments passed to the program.
        """
        self.parent.save_examples_used(model, args)
