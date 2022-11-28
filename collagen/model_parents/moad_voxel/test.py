import glob
from argparse import ArgumentParser, Namespace
import cProfile
from io import StringIO
import json
import pstats
import os
import re
from collagen.core.loader import DataLambda
import torch
from tqdm.std import tqdm
from typing import Any, List, Optional, Tuple

from collagen.core.molecules.mol import mols_from_smi_file
from collagen.core.voxelization.voxelizer import VoxelParams
from collagen.external.moad.types import Entry_info, MOAD_split
from collagen.metrics.ensembled import averaged as ensemble_helper
from collagen.external.moad.interface import MOADInterface, PfizerInterface
from collagen.external.moad.split import compute_moad_split
from collagen.metrics.metrics import (
    most_similar_matches,
    pca_space_from_label_set_fingerprints,
    top_k,
)


class MoadVoxelModelTest(object):

    @staticmethod
    def _remove_redundant_fingerprints(
            label_set_fps: torch.Tensor, label_set_smis: List[str], device: Any
    ) -> Tuple[torch.Tensor, List[str]]:
        # Given ordered lists of fingerprints and smiles strings, removes
        # redundant fingerprints and smis while maintaining the consistent order
        # between the two lists.
        label_set_fps, inverse_indices = label_set_fps.unique(
            dim=0, return_inverse=True
        )

        label_set_smis = [
            inf[1]
            for inf in sorted(
                [
                    (inverse_idx, label_set_smis[smi_idx])
                    for inverse_idx, smi_idx in {
                        int(inverse_idx): smi_idx
                        for smi_idx, inverse_idx in enumerate(inverse_indices)
                    }.items()
                ]
            )
        ]

        return label_set_fps, label_set_smis

    def _add_fingerprints_to_label_set_tensor(
        self: "MoadVoxelModelParent",
        args: Namespace,
        moad: MOADInterface,
        split: MOAD_split,
        voxel_params: VoxelParams,
        device: Any,
        existing_label_set_fps: torch.Tensor,
        existing_label_set_smis: List[str],
    ):
        # When testing a model, it's sometimes helpful to add additional
        # fingerprints to the label set (lookup table), beyond those in the test
        # set. This function allows you to include additional fingerprints
        # (e.g., from the training and validation sets) in the label set for
        # testing.

        # TODO: Harrison: How hard would it be to make it so data below doesn't
        # voxelize the receptor? Is that adding a lot of time to the
        # calculation? Just a thought. See other TODO: note about this.
        data = self.get_data_from_split(
            args, moad, split, voxel_params, device, shuffle=False
        )

        all_fps = []
        all_smis = []
        for batch in tqdm(data, desc=f"Getting fingerprints from {split.name} set..."):
            voxels, fps_tnsr, smis = batch
            all_fps.append(fps_tnsr)
            all_smis.extend(smis)
        all_smis.extend(existing_label_set_smis)
        all_fps.append(existing_label_set_fps)
        fps_tnsr = torch.cat(all_fps)

        # Remove redundancies.
        fps_tnsr, all_smis = self._remove_redundant_fingerprints(
            fps_tnsr, all_smis, device
        )

        return fps_tnsr, all_smis

    def _create_label_set_tensor(
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
    ) -> torch.Tensor:
        # Creates a label set (look-up) tensor for testing. Can be comprised of
        # the fingerprints in the train and/or test and/or val sets, as well as
        # SMILES strings from a file.

        # skip_test_set can be true if those fingerprints are already in
        # existing_label_set

        if lbl_set_codes is None:
            lbl_set_codes = [p.strip() for p in args.inference_label_sets.split(",")]

        if existing_label_set_fps is None:
            label_set_fps = torch.zeros((0, args.fp_size), device=device)
            label_set_smis = []
        else:
            # If you get an existing set of fingerprints, be sure to keep only the
            # unique ones.
            label_set_fps, label_set_smis = self._remove_redundant_fingerprints(
                existing_label_set_fps, existing_label_set_entry_infos, device=device
            )

        # Load from train, valm and test sets.
        if "train" in lbl_set_codes:
            label_set_fps, label_set_smis = self._add_fingerprints_to_label_set_tensor(
                args, moad, train, voxel_params, device, label_set_fps, label_set_smis
            )

        if "val" in lbl_set_codes:
            label_set_fps, label_set_smis = self._add_fingerprints_to_label_set_tensor(
                args, moad, val, voxel_params, device, label_set_fps, label_set_smis
            )

        if "test" in lbl_set_codes and not skip_test_set:
            label_set_fps, label_set_smis = self._add_fingerprints_to_label_set_tensor(
                args, moad, test, voxel_params, device, label_set_fps, label_set_smis
            )

        # Add to that fingerprints from an SMI file.
        smi_files = [f for f in lbl_set_codes if f not in ["train", "val", "test"]]
        if len(smi_files) > 0:
            fp_tnsrs_from_smi_file = [label_set_fps]
            for filename in smi_files:
                for smi, mol in mols_from_smi_file(filename):
                    fp_tnsrs_from_smi_file.append(
                        torch.tensor(
                            mol.fingerprint("rdk10", args.fp_size), device=device
                        ).reshape((1, args.fp_size))
                    )
                    label_set_smis.append(smi)
            label_set_fps = torch.cat(fp_tnsrs_from_smi_file)

            # Remove redundancy
            label_set_fps, label_set_smis = self._remove_redundant_fingerprints(
                label_set_fps, label_set_smis, device
            )

        # debug_smis_match_fps(label_set_fps, label_set_smis, device)

        print(f"Label set size: {len(label_set_fps)}")

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
    ) -> Any:
        # Certain variables can only be defined when processing the first
        # checkpoint. Note that all_test_data is modified in place and so does
        # not need to be returned.

        voxel_params = self.init_voxel_params(args)

        # Get the label set to use. Note that it only does this once (for the
        # first-checkpoint model), but I need model so I'm leaving it in the
        # loop.
        label_set_fingerprints, label_set_entry_infos = self._create_label_set_tensor(
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

        avg_over_ckpts_of_avgs = torch.zeros(
            model.prediction_targets.shape, device=device
        )

        return (
            pca_space,
            avg_over_ckpts_of_avgs,
            label_set_fingerprints,
            label_set_entry_infos,
        )

    def _run_test_on_checkpoint(
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
        # Note that you're iterating through multiple checkpoints. This allows
        # output from multiple trained models to be averaged. This is the test
        # run on a single checkpoint.

        # all_test_data is modified in place and so does not need to be
        # returned.

        # Could pass these as parameters, but let's keep things simple and just
        # reinitialize.
        device = self.init_device(args)

        predictions_per_rot = ensemble_helper.AveragedEnsembled(
            trainer, model, test_data, args.inference_rotations, device, ckpt, args.aggregation_rotations
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
        label_set_fingerprints = torch.tensor(label_set_fingerprints, dtype=torch.float32, device=device, requires_grad=True)
        top_k_results = top_k(
            predictions_ensembled,
            torch.tensor(model.prediction_targets, dtype=torch.float32, device=device, requires_grad=True),
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
            self.NUM_MOST_SIMILAR_PER_ENTRY,
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

    @staticmethod
    def _save_test_results_to_json(all_test_data, s, args, pth=None):
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
        pth = pth + os.sep + args.aggregation_rotations + os.sep
        os.makedirs(pth, exist_ok=True)
        num = len(glob.glob(pth + "*.json", recursive=False))

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

    def run_test(self: "MoadVoxelModelParent", args: Namespace, ckpt: Optional[str]):
        # Runs a model on the test and evaluates the output.

        pr = cProfile.Profile()
        pr.enable()

        if not ckpt:
            raise ValueError("Must specify a checkpoint in test mode")

        lbl_set_codes = [p.strip() for p in args.inference_label_sets.split(",")]

        if "test" not in lbl_set_codes:
            raise ValueError(
                "To run in test mode, you must include the `test` label set"
            )

        voxel_params = self.init_voxel_params(args)
        device = self.init_device(args)

        if args.csv and args.data:
            print("Test mode on the MOAD test dataset. Using the operator " + args.aggregation_rotations + " to aggregate the inferences.")
            moad = MOADInterface(
                metadata=args.csv,
                structures=args.data,
                cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                grid_width=voxel_params.width,
                grid_resolution=voxel_params.resolution,
                noh=args.noh,
                discard_distant_atoms=args.discard_distant_atoms,
            )
        elif not args.csv and args.data:
            print("Test mode on a test dataset other than the MOAD test dataset. Using the operator " + args.aggregation_rotations + " to aggregate the inferences.")
            moad = PfizerInterface(
                structures=args.data,
                cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                grid_width=voxel_params.width,
                grid_resolution=voxel_params.resolution,
                noh=args.noh,
                discard_distant_atoms=args.discard_distant_atoms,
            )
        elif args.csv and not args.data:
            raise Exception("To run the test mode on the MOAD test database is required to specify the --csv and --data arguments")
        elif not args.csv and not args.data:
            raise Exception("To run the test mode is required to specify the --csv and --data arguments (for MOAD database), or the --data argument only (for a database other than MOAD)")

        train, val, test = compute_moad_split(
            moad,
            args.split_seed,
            save_splits=args.save_splits,
            load_splits=args.load_splits,
            max_pdbs_train=args.max_pdbs_train,
            max_pdbs_val=args.max_pdbs_val,
            max_pdbs_test=args.max_pdbs_test,
            prevent_smiles_overlap=False,  # DEBUG
        )

        # You'll always need the test data. Note that ligands are not fragmented
        # by calling the get_data_from_split function.
        test_data = self.get_data_from_split(
            args, moad, test, voxel_params, device, shuffle=False
        )

        trainer = self.init_trainer(args)

        ckpts = [c.strip() for c in ckpt.split(",")]
        all_test_data = {
            "checkpoints": [{"name": c, "order": i + 1} for i, c in enumerate(ckpts)],
            "entries": [],
        }
        avg_over_ckpts_of_avgs = None
        for ckpt_idx, ckpt in enumerate(ckpts):
            # You're iterating through multiple checkpoints. This allows output
            # from multiple trained models to be averaged.

            model = self.init_model(args, ckpt)
            model.eval()

            payload = self._run_test_on_checkpoint(
                all_test_data,
                args,
                model,
                ckpt_idx,
                ckpt,
                trainer,
                test_data,
                train,
                val,
                moad,
                lbl_set_codes,
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
        label_set_fingerprints = torch.tensor(label_set_fingerprints, dtype=torch.float32, device=device, requires_grad=True)
        top_k_results = top_k(
            avg_over_ckpts_of_avgs,
            torch.tensor(model.prediction_targets, dtype=torch.float32, device=device, requires_grad=True),
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
            self.NUM_MOST_SIMILAR_PER_ENTRY,
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
