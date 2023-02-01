import glob
from argparse import ArgumentParser, Namespace
import cProfile
from io import StringIO
import json
import pstats
import os
import re
from collagen.core.loader import DataLambda
from collagen.model_parents.moad_voxel.test_inference_utils import remove_redundant_fingerprints
import torch
from tqdm.std import tqdm
from typing import Any, List, Optional, Tuple
# import multiprocessing
# from torch import multiprocessing
from collagen.core.molecules.mol import mols_from_smi_file
from collagen.core.voxelization.voxelizer import VoxelParams
from collagen.external.moad.types import Entry_info, MOAD_split
from collagen.metrics.ensembled import averaged as ensemble_helper
from collagen.external.moad.interface import MOADInterface, PdbSdfDirInterface
from collagen.external.moad.split import compute_dataset_split
from collagen.metrics.metrics import (
    most_similar_matches,
    pca_space_from_label_set_fingerprints,
    top_k,
)

# See https://github.com/pytorch/pytorch/issues/3492
# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass
# multiprocessing_ctx = multiprocessing.get_context("spawn")

def _return_paramter(object):
    """Returns a paramerter. For use in imap_unordered.

    Args:
        object (Any): The parameter.

    Returns:
        Any: The parameter returned.
    """
    return object


class MoadVoxelModelTest(object):
    def __init__(self, model_parent):
        self.model_parent = model_parent

    def _add_to_label_set(
        self: "MoadVoxelModelParent",
        args: Namespace,
        moad: MOADInterface,
        split: MOAD_split,
        voxel_params: VoxelParams,
        device: Any,
        existing_label_set_fps: torch.Tensor,
        existing_label_set_smis: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:
        """This is a helper script. Adds fingerprints to a label set (lookup
        table). This function allows you to include additional fingerprints
        (e.g., from the training and validation sets) in the label set for
        testing. Takes the fingerprints right from the split.
        
        Args:
            self (MoadVoxelModelParent): This object
            args (Namespace): The user arguments.
            moad (MOADInterface): The MOAD dataset.
            split (MOAD_split): The splits of the MOAD dataset.
            voxel_params (VoxelParams): Parameters for voxelization.
            device (Any): The device to use.
            existing_label_set_fps (torch.Tensor): The existing tensor of 
                fingerprints to which these new ones should be added.
            existing_label_set_smis (List[str]): The existing list of SMILES
                strings to which the new ones should be added.

        Returns:
            Tuple[torch.Tensor, List[str]]: The updated fingerprint
                tensor and smiles list.
        """

        # global multiprocessing_ctx

        # TODO: Harrison: How hard would it be to make it so data below doesn't
        # voxelize the receptor? Is that adding a lot of time to the
        # calculation? Just a thought. See other TODO: note about this.
        data = self.model_parent.get_data_from_split(
            cache_file=args.cache, 
            args=args, 
            dataset=moad, 
            split=split, 
            voxel_params=voxel_params, 
            device=device, 
            shuffle=False,
        )

        all_fps = []
        all_smis = []

        # with multiprocessing.Pool() as p:
        #     for batch in tqdm(
        #         p.imap_unordered(_return_paramter, data), 
        #         total=len(data), 
        #         desc=f"Getting fingerprints from {split.name if split else 'Full'} set..."
        #     ):
        #         voxels, fps_tnsr, smis = batch
        #         all_fps.append(fps_tnsr)
        #         all_smis.extend(smis)

        # The above causes CUDA errors. Unfortunate, because it would speed
        # things up quite a bit. TODO: DISCUSS WITH CESAR. Good to look into this.
        for batch in tqdm(data, desc=f"Getting fingerprints from {split.name if split else 'Full'} set..."):
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
        self: "MoadVoxelModelParent",
        args: ArgumentParser,
        device: Any,
        existing_label_set_fps: torch.Tensor = None,
        existing_label_set_entry_infos: List[Entry_info] = None,
        skip_test_set=False,
        train: MOAD_split = None,
        val: MOAD_split = None,
        test: MOAD_split = None,
        moad: MOADInterface = None,
        voxel_params: VoxelParams = None,
        lbl_set_codes: List[str] = None,
    ) -> Tuple[torch.Tensor, List[Entry_info]]:
        """Creates a label set (look-up) tensor and smiles list for testing. 
        Can be comprised of the fingerprints in the train and/or test and/or
        val sets, as well as SMILES strings from a file.

        Args:
            self (MoadVoxelModelParent): This object
            args (Namespace): The user arguments.
            device (Any): The device to use.
            existing_label_set_fps (torch.Tensor, optional): The existing tensor of 
                fingerprints to which these new ones should be added. Defaults
                to None.
            existing_label_set_entry_infos (List[Entry_info], optional): _description_. Defaults to None.
            skip_test_set (bool, optional): Do not add test-set fingerprints,
                presumably because they are already present in
                existing_label_set_entry_infos. Defaults to False.
            train (MOAD_split, optional): The train split. Defaults to None.
            val (MOAD_split, optional): The val split. Defaults to None.
            test (MOAD_split, optional): The test split. Defaults to None.
            moad (MOADInterface, optional): The MOAD dataset. Defaults to None.
            voxel_params (VoxelParams): Parameters for voxelization. Defaults to None.
            lbl_set_codes (List[str], optional): _description_. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[Entry_info]]: The updated fingerprint
                tensor and smiles list.
        """

        if "all" in args.inference_label_sets:
            raise Exception("The 'all' value for the --inference_label_sets parameter is not a valid value in test mode")

        # skip_test_set can be true if those fingerprints are already in
        # existing_label_set

        if lbl_set_codes is None:
            lbl_set_codes = [p.strip() for p in args.inference_label_sets.split(",")]

        # Load from train, val, and test sets.
        if existing_label_set_fps is None:
            label_set_fps = torch.zeros(
                (0, args.fp_size), dtype=torch.float32, device=device, requires_grad=False
            )
            label_set_smis = []
        else:
            # If you get an existing set of fingerprints, be sure to keep only the
            # unique ones.
            label_set_fps, label_set_smis = remove_redundant_fingerprints(
                existing_label_set_fps, existing_label_set_entry_infos, device=device
            )

        if "train" in lbl_set_codes and len(train.targets) > 0:
            label_set_fps, label_set_smis = self._add_to_label_set(
                args, moad, train, voxel_params, device, label_set_fps, label_set_smis
            )

        if "val" in lbl_set_codes and len(val.targets) > 0:
            label_set_fps, label_set_smis = self._add_to_label_set(
                args, moad, val, voxel_params, device, label_set_fps, label_set_smis
            )

        if "test" in lbl_set_codes and not skip_test_set and len(test.targets) > 0:
            label_set_fps, label_set_smis = self._add_to_label_set(
                args, moad, test, voxel_params, device, label_set_fps, label_set_smis
            )

        # Add to that fingerprints from an SMI file.
        label_set_fps, label_set_smis = self._add_fingerprints_from_smis(args, lbl_set_codes, label_set_fps, label_set_smis, device)

        # self.model_parent.debug_smis_match_fps(label_set_fps, label_set_smis, device, args)

        print(f"Label set size: {len(label_set_fps)}")

        return label_set_fps, label_set_smis

    def _add_fingerprints_from_smis(self, args, lbl_set_codes, label_set_fps, label_set_smis, device):
        smi_files = [f for f in lbl_set_codes if f not in ["train", "val", "test", "all"]]
        if smi_files:
            fp_tnsrs_from_smi_file = [label_set_fps]
            for filename in smi_files:
                for smi, mol in mols_from_smi_file(filename):
                    fp_tnsrs_from_smi_file.append(
                        torch.tensor(
                            mol.fingerprint(args.molecular_descriptors, args.fp_size),
                            dtype=torch.float32, device=device,
                            requires_grad=False
                        ).reshape((1, args.fp_size))
                    )
                    label_set_smis.append(smi)
            label_set_fps = torch.cat(fp_tnsrs_from_smi_file)

            # Remove redundancy
            label_set_fps, label_set_smis = remove_redundant_fingerprints(
                label_set_fps, label_set_smis, device
            )

        return label_set_fps, label_set_smis

    def _on_first_checkpoint(
        self: "MoadVoxelModelParent",
        all_test_data: Any,
        args: Namespace,
        model: Any,
        train: MOAD_split,
        val: MOAD_split,
        moad: MOADInterface,
        lbl_set_codes: List[str],
        device: Any,
        predictions_per_rot: ensemble_helper.AveragedEnsembled,
    ) -> Tuple["PCAProject", torch.Tensor, torch.Tensor, List[Entry_info]]:
        """Certain variables can only be defined when processing the first
        checkpoint. Moving this out of the loop so it's not distracting. Note
        that all_test_data is modified in place and so does not need to be
        returned."""

        voxel_params = self.model_parent.init_voxel_params(args)

        # Get the label set to use. Note that it only does this once (for the
        # first-checkpoint model), but I need model so I'm leaving it in the
        # loop.
        label_set_fingerprints, label_set_entry_infos = self._create_label_set(
            args,
            device,
            existing_label_set_fps=predictions_per_rot.model.prediction_targets,
            existing_label_set_entry_infos=predictions_per_rot.model.prediction_targets_entry_infos,
            skip_test_set=True,
            train=train,
            val=val,
            moad=moad,
            voxel_params=voxel_params,
            lbl_set_codes=lbl_set_codes,
        )

        # Get a PCA (or other) space defined by the label-set fingerprints.
        pca_space = pca_space_from_label_set_fingerprints(label_set_fingerprints, 2)

        all_test_data["pcaPercentVarExplainedByEachComponent"] = [
            100 * r for r in pca_space.pca.explained_variance_ratio_.tolist()
        ]

        return (
            pca_space,
            torch.zeros(model.prediction_targets.shape, dtype=torch.float32, device=device, requires_grad=False),
            torch.tensor(label_set_fingerprints, dtype=torch.float32, device=device, requires_grad=False),
            label_set_entry_infos,
        )

    def _run_test_on_single_checkpoint(
        self: "MoadVoxelModelParent",
        all_test_data: Any,
        args: Namespace,
        model: Any,
        ckpt_idx: int,
        ckpt: str,
        trainer: Any,
        test_data: DataLambda,
        train: MOAD_split,
        val: MOAD_split,
        moad: MOADInterface,
        lbl_set_codes: List[str],
        avg_over_ckpts_of_avgs: Any,
    ) -> Any:
        """This is the test run on a single checkpoint. You're iterating through
        multiple checkpoints. This allows output from multiple trained models to
        be averaged."""

        # all_test_data is modified in place and so does not need to be
        # returned.

        # Could pass these as parameters, but let's keep things simple and just
        # reinitialize.
        device = self.model_parent.init_device(args)

        predictions_per_rot = ensemble_helper.AveragedEnsembled(
            trainer, model, test_data, args.rotations, device, ckpt, args.aggregation_rotations
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
                moad,
                lbl_set_codes,
                device,
                predictions_per_rot,
            )

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
                    {"correct": correct_answer, "perCheckpoint": []}
                )

        # Calculate top_k metric for this checkpoint
        top_k_results = top_k(
            predictions_ensembled,
            torch.tensor(model.prediction_targets, dtype=torch.float32, device=device, requires_grad=False),
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
            self.model_parent.NUM_MOST_SIMILAR_PER_ENTRY,
            pca_space,
        )
        for entry_idx in range(len(predictions_ensembled)):
            # Add closest compounds from label set.
            for predicted_entry_info, dist, pca in most_similar[entry_idx]:
                all_test_data["entries"][entry_idx]["perCheckpoint"][-1][
                    "averagedPrediction"
                ]["closestFromLabelSet"].append(
                    {
                        "smiles": predicted_entry_info.fragment_smiles,
                        "cosineDistToAveraged": dist,
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
        else:
            return None

    def _save_test_results_to_json(self, all_test_data, s, args, pth=None):
        # Save the test results to a carefully formatted JSON file.

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
        pth = pth + os.sep + self._get_json_name(args) + os.sep + args.aggregation_rotations + os.sep
        os.makedirs(pth, exist_ok=True)
        num = len(glob.glob(f"{pth}*.json", recursive=False))

        with open(f"{pth}test_results-{num + 1}.json", "w") as f:
            f.write(jsn)
        with open(f"{pth}cProfile-{num + 1}.txt", "w+") as f:
            f.write(s.getvalue())

        # txt = ""
        # for entry in all_test_data["entries"]:
        #     txt += "Correct\n"
        #     txt += "\t".join([str(e) for e in entry["correct"]["pcaProjection"]]) + "\t" + entry["correct"]["fragmentSmiles"] + "\n"
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

    def _validate_run_test(self: "MoadVoxelModelParent", args: Namespace, ckpt: Optional[str]):
        if not ckpt:
            raise ValueError("Must specify a checkpoint in test mode")
        elif not args.inference_label_sets:
            raise ValueError("Must specify a label set (--inference_label_sets argument)")
        elif args.every_csv and not args.data_dir:
            raise Exception("To load the MOAD database, you must specify the --every_csv and --data_dir arguments")
        elif not args.every_csv and not args.data_dir:
            raise Exception("To run the test mode, you must specify the --every_csv and --data_dir arguments (for MOAD database), or the --data_dir argument only for a database other than MOAD")
        elif args.custom_test_set_dir:
            raise Exception("To run the test mode must not be specified a custom dataset (--custom_test_set_dir argument)")
        elif "all" in args.inference_label_sets:
            raise Exception("The `all` value for label set (--inference_label_sets) must not be specified in test mode")
        elif "test" not in args.inference_label_sets:
            raise ValueError("To run in test mode, you must include the `test` label set")
        elif not args.load_splits:
            raise Exception("To run the test mode is required loading a previously saved test dataset")

    def run_test(
        self: "MoadVoxelModelParent", args: Namespace, ckpt: Optional[str]
    ):
        # Runs a model on the test and evaluates the output.

        pr = cProfile.Profile()
        pr.enable()

        self._validate_run_test(args, ckpt)

        print(f"Using the operator {args.aggregation_rotations} to aggregate the inferences.")

        voxel_params = self.model_parent.init_voxel_params(args)
        device = self.model_parent.init_device(args)

        dataset, set2run_test_on_single_checkpoint = self._read_datasets2run_test(args, voxel_params)

        train, val, test = compute_dataset_split(
            dataset=dataset,
            seed=None,
            fraction_train=0.0,
            fraction_val=0.0,
            prevent_smiles_overlap=False,  # DEBUG
            save_splits=None,
            load_splits=self._get_load_splits(args),
            max_pdbs_train=args.max_pdbs_train,
            max_pdbs_val=args.max_pdbs_val,
            max_pdbs_test=args.max_pdbs_test,
            butina_cluster_division=False,
            butina_cluster_cutoff=0.0,
        )

        # You'll always need the test data. Note that ligands are not fragmented
        # by calling the get_data_from_split function.
        test_data = self.model_parent.get_data_from_split(
            cache_file=self._get_cache(args),
            args=args,
            dataset=dataset,
            split=test, 
            voxel_params=voxel_params, 
            device=device, 
            shuffle=False,
        )
        print(f"Number of batches for the test data: {len(test_data)}")

        trainer = self.model_parent.init_trainer(args)

        ckpts = [c.strip() for c in ckpt.split(",")]
        all_test_data = {
            "checkpoints": [{"name": c, "order": i + 1} for i, c in enumerate(ckpts)],
            "entries": [],
        }
        avg_over_ckpts_of_avgs = None
        for ckpt_idx, ckpt in enumerate(ckpts):
            # You're iterating through multiple checkpoints. This allows output
            # from multiple trained models to be averaged.

            model = self.model_parent.init_model(args, ckpt)
            model.eval()
            
            # TODO: model.device is "cpu". Is that right? Shouldn't it be "cuda"?

            payload = self._run_test_on_single_checkpoint(
                all_test_data,
                args,
                model,
                ckpt_idx,
                ckpt,
                trainer,
                test_data,
                train,
                val,
                set2run_test_on_single_checkpoint,
                None,
                avg_over_ckpts_of_avgs,
            )

            if ckpt_idx == 0:
                # Payload is not None if first checkpoint.
                (
                    pca_space,
                    avg_over_ckpts_of_avgs,
                    label_set_fingerprints,
                    label_set_entry_infos,
                ) = payload

        # Get the average of averages (across all checkpoints)
        torch.div(
            avg_over_ckpts_of_avgs,
            torch.tensor(len(ckpts), device=device),
            out=avg_over_ckpts_of_avgs,
        )

        # Calculate top-k metric of that average of averages
        top_k_results = top_k(
            avg_over_ckpts_of_avgs,
            torch.tensor(model.prediction_targets, dtype=torch.float32, device=device, requires_grad=False),
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
            self.model_parent.NUM_MOST_SIMILAR_PER_ENTRY,
            pca_space,
        )
        for entry_idx in range(len(avg_over_ckpts_of_avgs)):
            # Add closest compounds from label set.
            all_test_data["entries"][entry_idx]["avgOfCheckpoints"] = {
                "pcaProjection": avg_over_ckpts_of_avgs_viz[entry_idx],
                "closestFromLabelSet": [],
            }
            for predicted_entry_info, dist, pca in most_similar[entry_idx]:
                all_test_data["entries"][entry_idx]["avgOfCheckpoints"][
                    "closestFromLabelSet"
                ].append(
                    {
                        "smiles": predicted_entry_info.fragment_smiles,
                        "cosineDistToAveraged": dist,
                        "pcaProjection": pca[0],
                    }
                )

        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()

        self._save_test_results_to_json(all_test_data, s, args, args.default_root_dir)
        self._save_examples_used(model, args)

    def _read_datasets2run_test(self, args, voxel_params):
        if args.every_csv and args.data_dir:
            print("Loading MOAD database.")
            dataset = self._read_BidingMOAD_database(args, voxel_params)
        else:
            print("Loading a database other than MOAD database.")
            dataset = PdbSdfDirInterface(
                structures=args.data_dir,
                cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                grid_width=voxel_params.width,
                grid_resolution=voxel_params.resolution,
                noh=args.noh,
                discard_distant_atoms=args.discard_distant_atoms,
            )

        return dataset, dataset

    def _read_BidingMOAD_database(self, args, voxel_params):
        moad = MOADInterface(
            metadata=args.every_csv,
            structures_path=args.data_dir,
            cache_pdbs_to_disk=args.cache_pdbs_to_disk,
            grid_width=voxel_params.width,
            grid_resolution=voxel_params.resolution,
            noh=args.noh,
            discard_distant_atoms=args.discard_distant_atoms,
        )
        return moad

    def _get_load_splits(self, args):
        return args.load_splits

    def _get_cache(self, args):
        return args.cache

    def _get_json_name(self, args):
        return "predictions_MOAD" if (args.data_dir and args.every_csv) else "predictions_nonMOAD"

    def _save_examples_used(self, model, args):
        self.model_parent.save_examples_used(model, args)
