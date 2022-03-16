import argparse
from functools import partial
from multiprocessing import Value, cpu_count
from typing import Any, Type, TypeVar, List, Optional, Tuple
from collagen.core.loader import DataLambda
from collagen.core.mol import Mol, mols_from_smi_file
from collagen.external.moad.types import Entry_info, MOAD_split
from tqdm.std import tqdm
import cProfile
import pstats
from io import StringIO
import numpy as np
from collagen.metrics.ensembled import averaged as ensemble_helper
# from collagen.metrics.ensembled import clustered as ensemble_helper

import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
import torch

from ..checkpoints import MyModelCheckpoint, get_last_checkpoint
from .. import VoxelParams, VoxelParamsDefault, MultiLoader
from ..external import MOADInterface

from collagen.metrics import make_vis_rep_space_from_label_set_fingerprints, most_similar_matches, top_k

import re
import json


ENTRY_T = TypeVar("ENTRY_T")
TMP_T = TypeVar("TMP_T")
OUT_T = TypeVar("OUT_T")

NUM_MOST_SIMILAR_PER_ENTRY = 5


def _disable_warnings():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    import prody

    prody.confProDy(verbosity="none")


class MoadVoxelSkeleton(object):
    def __init__(
        self,
        model_cls: Type[pl.LightningModule],
        dataset_cls: Type[torch.utils.data.Dataset]
    ):
        self.model_cls = model_cls
        self.dataset_cls = dataset_cls

    @staticmethod
    def add_moad_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("Binding MOAD")

        parser.add_argument("--csv", required=True, help="Path to MOAD every.csv")
        parser.add_argument(
            "--data", required=True, help="Path to MOAD root structure folder"
        )
        parser.add_argument(
            "--cache",
            required=False,
            default=None,
            help="Path to MOAD cache.json file. If not given, `.cache.json` is appended to the file path given by `--csv`.",
        )
        parser.add_argument(
            "--cache_pdbs",
            default=False,
            action="store_true",
            help="If given, collagen will convert the PDB files to a faster cachable format. Will run slower the first epoch, but faster on subsequent epochs and runs."
        )
        parser.add_argument(
            "--noh",
            default=False,
            action="store_true",
            help="If given, collagen will not use protein hydrogen atoms, nor will it save them to the cachable files generated with --cache_pdbs. Can speed calculations and free disk space if your model doesn't need hydrogens, and if you're using --cache_pdbs."
        )
        parser.add_argument(
            "--discard_distant_atoms",
            default=False,
            action="store_true",
            help="If given, collagen will not consider atoms that are far from any ligand, nor will it save them to the cachable files generated with --cache_pdbs. Can speed calculations and free disk space if you're using --cache_pdbs."
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
            "--max_pdbs_to_use_in_train",
            required=False,
            default=None,
            type=int,
            help="If given, the max number of PDBs used to generate examples in the train set. If this set contains more than `max_pdbs_to_use_in_train` PDBs, extra PDBs will be removed.",
        )
        parser.add_argument(
            "--max_pdbs_to_use_in_val",
            required=False,
            default=None,
            type=int,
            help="If given, the max number of PDBs used to generate examples in the val set. If this set contains more than `max_pdbs_to_use_in_val` PDBs, extra PDBs will be removed.",
        )
        parser.add_argument(
            "--max_pdbs_to_use_in_test",
            required=False,
            default=None,
            type=int,
            help="If given, the max number of PDBs used to generate examples in the test set. If this set contains more than `max_pdbs_to_use_in_test` PDBs, extra PDBs will be removed.",
        )

        parser.add_argument(
            "--num_dataloader_workers",
            default=cpu_count(),
            type=int,
            help="Number of workers for DataLoader",
        )
        parser.add_argument(
            "--max_voxels_in_memory",
            required=True,
            default=80,
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
            default=1,
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

    @staticmethod
    def fix_moad_args(args: argparse.Namespace) -> argparse.Namespace:
        # Only works after arguments have been parsed, so in a separate
        # definition.
        if args.cache is None:
            args.cache = args.csv + ".cache.json"
        return args

    @staticmethod
    def pre_voxelize(
        args: argparse.Namespace, voxel_params: VoxelParams, entry: ENTRY_T
    ) -> TMP_T:
        return entry

    @staticmethod
    def voxelize(
        args: argparse.Namespace,
        voxel_params: VoxelParams,
        device: torch.device,
        batch: List[TMP_T],
    ) -> OUT_T:
        raise NotImplementedError()

    @staticmethod
    def batch_eval(args: argparse.Namespace, batch: OUT_T):
        pass

    @staticmethod
    def custom_test(args: argparse.Namespace, predictions):
        pass

    def _init_trainer(self, args: argparse.Namespace) -> pl.Trainer:
        logger = None
        if args.wandb_project:
            logger = WandbLogger(project=args.wandb_project)
        else:
            logger = CSVLogger(
                "logs",
                name="my_exp_name",
                flush_logs_every_n_steps=args.log_every_n_steps,
            )

        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=[
                MyModelCheckpoint(
                    dirpath=args.default_root_dir,
                    monitor="val_loss",
                    filename="val-loss-{epoch:02d}-{val_loss:.2f}",
                    save_top_k=3,
                ),
                MyModelCheckpoint(
                    dirpath=args.default_root_dir,
                    monitor="loss",
                    filename="loss-{epoch:02d}-{loss:.2f}",
                    save_last=True,
                    save_top_k=3,
                ),
            ],
        )

        return trainer

    def _init_model(
        self, args: argparse.Namespace, ckpt: Optional[str]
    ) -> pl.LightningModule:
        if ckpt:
            print(f"\nLoading model from checkpoint {ckpt}\n")
            return self.model_cls.load_from_checkpoint(ckpt)
        else:
            return self.model_cls(**vars(args))

    def _init_voxel_params(self, args: argparse.Namespace) -> VoxelParams:
        # TODO: make configurable via argparse
        return VoxelParamsDefault.DeepFrag

    def _init_device(self, args: argparse.Namespace) -> torch.device:
        if args.cpu:
            return torch.device("cpu")
        else:
            return torch.device("cuda")

    def _get_checkpoint(self, args: argparse.Namespace) -> Optional[str]:
        if args.load_checkpoint and args.load_newest_checkpoint:
            raise ValueError(
                f"Can specify 'load_checkpoint=xyz' or 'load_newest_checkpoint' but not both."
            )

        ckpt = None
        if args.load_checkpoint:
            ckpt = args.load_checkpoint
        elif args.load_newest_checkpoint:
            ckpt = get_last_checkpoint(args)

        return ckpt

    def run(self, args: argparse.Namespace = None):
        _disable_warnings()

        ckpt = self._get_checkpoint(args)
        if ckpt is not None:
            print(f"Restoring from checkpoint: {ckpt}")

        if args.mode == "train":
            self._run_train(args, ckpt)
        elif args.mode == "test":
            self._run_test(args, ckpt)
        elif args.mode == "lr_finder":
            self._run_lr_finder(args)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    def _get_data_from_split(
        self, args: argparse.Namespace, moad: MOADInterface, split: MOAD_split, 
        voxel_params: VoxelParams, device: Any, shuffle=True
    ) -> DataLambda:
        # This is where you do actual dataset construction. The transform
        # function actually gets the data (voxelizes and creates fingerprint).
        # TODO: Create separate function .pre_voxelize_with_voxel that just
        # calculates just fingerprint.

        dataset = self.dataset_cls(
            moad,
            cache_file=args.cache,
            split=split,
            transform=(lambda entry: self.__class__.pre_voxelize(args, voxel_params, entry)),
            args=args
        )
        data = (
            MultiLoader(
                dataset,
                shuffle=shuffle,
                num_dataloader_workers=args.num_dataloader_workers,
                max_voxels_in_memory=args.max_voxels_in_memory,
            )
            .batch(args.batch_size)
            .map(lambda batch: self.__class__.voxelize(args, voxel_params, device, batch))
        )

        return data

    def _run_lr_finder(self, args: argparse.Namespace):
        # Update value of auto_lr_find
        args = vars(args)
        args["auto_lr_find"] = True
        args = argparse.Namespace(**args)

        trainer = self._init_trainer(args)
        model = self._init_model(args, None)
        voxel_params = self._init_voxel_params(args)
        device = self._init_device(args)

        moad = MOADInterface(
            metadata=args.csv, structures=args.data, cache_pdbs=args.cache_pdbs,
            grid_width=voxel_params.width, grid_resolution=voxel_params.resolution,
            noh=args.noh, discard_distant_atoms=args.discard_distant_atoms
        )
        train, val, _ = moad.compute_split(
            args.split_seed, save_splits=args.save_splits,
            load_splits=args.load_splits,
            max_pdbs_to_use_in_train=args.max_pdbs_to_use_in_train,
            max_pdbs_to_use_in_val=args.max_pdbs_to_use_in_val,
            max_pdbs_to_use_in_test=args.max_pdbs_to_use_in_test,
        )

        train_data = self._get_data_from_split(
            args, moad, train, voxel_params, device
        )

        val_data = self._get_data_from_split(
            args, moad, val, voxel_params, device
        )

        lr_finder = trainer.tuner.lr_find(model, train_data, val_data)
        print("Suggested learning rate:", lr_finder.suggestion())

        # import pdb; pdb.set_trace()

    def _run_train(self, args: argparse.Namespace, ckpt: Optional[str]):
        trainer = self._init_trainer(args)
        model = self._init_model(args, ckpt)
        voxel_params = self._init_voxel_params(args)
        device = self._init_device(args)

        moad = MOADInterface(
            metadata=args.csv, structures=args.data, cache_pdbs=args.cache_pdbs, 
            grid_width=voxel_params.width, grid_resolution=voxel_params.resolution,
            noh=args.noh, discard_distant_atoms=args.discard_distant_atoms
        )
        train, val, _ = moad.compute_split(
            args.split_seed, save_splits=args.save_splits,
            load_splits=args.load_splits,
            max_pdbs_to_use_in_train=args.max_pdbs_to_use_in_train,
            max_pdbs_to_use_in_val=args.max_pdbs_to_use_in_val,
            max_pdbs_to_use_in_test=args.max_pdbs_to_use_in_test,
        )

        train_data = self._get_data_from_split(
            args, moad, train, voxel_params, device
        )

        val_data = self._get_data_from_split(
            args, moad, val, voxel_params, device
        )

        trainer.fit(model, train_data, val_data, ckpt_path=ckpt)

    def _add_fingerprints_to_label_set_tensor(
        self, args: argparse.Namespace, moad: MOADInterface, split: MOAD_split, 
        voxel_params: VoxelParams, device: Any, existing_label_set_fps: torch.Tensor,
        existing_label_set_smis: List[str]
    ):
        # If you want to include training and validation fingerprints in the
        # label set for testing.

        # TODO: Harrison: How hard would it be to make it so data below
        # doesn't voxelize the receptor? Is that adding a lot of time to the
        # calculation? Just a thought. See other TODO: note about this.
        data = self._get_data_from_split(
            args, moad, split, voxel_params, device, shuffle=False
        )

        all_fps = []
        all_smis = []
        for batch in tqdm(data, desc="Getting fingerprints from " + split.name + " set..."):
            voxels, fps_tnsr, smis = batch
            all_fps.append(fps_tnsr)
            all_smis.extend(smis)
        all_smis.extend(existing_label_set_smis)
        all_fps.append(existing_label_set_fps)
        fps_tnsr = torch.cat(all_fps)

        # Remove redundancies.
        fps_tnsr, all_smis = self._remove_redundant_fingerprints(fps_tnsr, all_smis, device)

        return fps_tnsr, all_smis

        # existing_label_set_fps, existing_label_set_smis = self._remove_redundant_fingerprints(
        #     existing_label_set_fps, existing_label_set_smis, device=device
        # )


        # # Note: This would only work if considering whole ligand. Perhaps
        # # good for deeplig, but not deepfrag.
        # mols = [Mol.from_smiles(s) for s in split.smiles]
        # fps = [
        #     torch.tensor(m.fingerprint("rdk10", args.fp_size), device=device).reshape((1, args.fp_size))
        #     for m in mols
        # ]
        # fps.append(label_set)
        # return torch.cat(fps).unique(dim=0)

    def _create_label_set_tensor(
        self, args: argparse.ArgumentParser, device: Any, 
        existing_label_set_fps: torch.Tensor=None,
        existing_label_set_entry_infos: List[Entry_info]=None,
        skip_test_set=False,
        train: MOAD_split = None, val: MOAD_split = None, 
        test: MOAD_split = None, moad: MOADInterface = None,
        voxel_params: VoxelParams = None, lbl_set_codes: List[str] = None
    ) -> torch.Tensor:
        # skip_test_set can be true if those fingerprints are already in
        # existing_label_set

        if lbl_set_codes is None:
            lbl_set_codes = [p.strip() for p in args.inference_label_sets.split(",")]

        if existing_label_set_fps is None:
            label_set_fps = torch.zeros((0, args.fp_size), device=device)
            label_set_smis = []
        else:
            # If you get an existing set of fingerprints, be sure to keep only
            # the unique ones.
            label_set_fps, label_set_smis = self._remove_redundant_fingerprints(
                existing_label_set_fps, existing_label_set_entry_infos, device=device
            )

        # Load from train, valm and test sets.
        if "train" in lbl_set_codes:
            label_set_fps, label_set_smis = self._add_fingerprints_to_label_set_tensor(
                args, moad, train, voxel_params, device, label_set_fps,
                label_set_smis
            )

        if "val" in lbl_set_codes:
            label_set_fps, label_set_smis = self._add_fingerprints_to_label_set_tensor(
                args, moad, val, voxel_params, device, label_set_fps,
                label_set_smis
            )

        if "test" in lbl_set_codes and not skip_test_set:
            label_set_fps, label_set_smis = self._add_fingerprints_to_label_set_tensor(
                args, moad, test, voxel_params, device, label_set_fps,
                label_set_smis
            )

        # Add to that fingerprints from an SMI file.
        smi_files = [f for f in lbl_set_codes if f not in ["train", "val", "test"]]
        if len(smi_files) > 0:
            fp_tnsrs_from_smi_file = [label_set_fps]
            for filename in smi_files:
                for smi, mol in mols_from_smi_file(filename):
                    fp_tnsrs_from_smi_file.append(
                        torch.tensor(
                            mol.fingerprint("rdk10", args.fp_size),
                            device=device
                        ).reshape((1, args.fp_size))
                    )
                    label_set_smis.append(smi)
            label_set_fps = torch.cat(fp_tnsrs_from_smi_file)
            
            # Remove redundancy
            label_set_fps, label_set_smis = self._remove_redundant_fingerprints(
                label_set_fps, label_set_smis, device
            )

        # self._debug_smis_match_fps(label_set_fps, label_set_smis, device)

        print('Label set size: ' + str(len(label_set_fps)))

        return label_set_fps, label_set_smis

    def _debug_smis_match_fps(self, fps: torch.Tensor, smis: List[str], device: Any):
        import rdkit
        from rdkit import Chem
        for idx in range(len(smis)):
            smi = smis[idx]
            fp1 = fps[idx]

            mol = Mol.from_smiles(smi)
            # TODO: 2048 should be hardcoded here? I think it's a user parameter.
            fp2 = torch.tensor(mol.fingerprint("rdk10", 2048), device=device, dtype=torch.float32)
            print((fp1-fp2).max() == (fp1-fp2).min())

        import pdb; pdb.set_trace()

    def _remove_redundant_fingerprints(
        self, label_set_fps: torch.Tensor, label_set_smis: List[str], device: Any
    )->Tuple[torch.Tensor, List[str]]:
        # Removes redundant fingerprints and smis, but maintaing the consistent
        # order between the two lists.
        label_set_fps, inverse_indices = label_set_fps.unique(dim=0, return_inverse=True)

        label_set_smis = [
            inf[1] for inf in sorted([
                (inverse_idx, label_set_smis[smi_idx]) 
                for inverse_idx, smi_idx in 
                {
                    int(inverse_idx): smi_idx 
                    for smi_idx, inverse_idx in 
                    enumerate(inverse_indices)
                }.items()
            ])
        ]
        
        return label_set_fps, label_set_smis

    def _run_test(self, args: argparse.Namespace, ckpt: Optional[str]):
        pr = cProfile.Profile()
        pr.enable()

        if not ckpt:
            raise ValueError("Must specify a checkpoint in test mode")

        lbl_set_codes = [p.strip() for p in args.inference_label_sets.split(",")]

        if not "test" in lbl_set_codes:
            raise ValueError("To run in test mode, you must include the `test` label set")

        voxel_params = self._init_voxel_params(args)
        device = self._init_device(args)

        moad = MOADInterface(
            metadata=args.csv, structures=args.data, cache_pdbs=args.cache_pdbs,
            grid_width=voxel_params.width, grid_resolution=voxel_params.resolution,
            noh=args.noh, discard_distant_atoms=args.discard_distant_atoms
        )
        train, val, test = moad.compute_split(
            args.split_seed, save_splits=args.save_splits,
            load_splits=args.load_splits,
            max_pdbs_to_use_in_train=args.max_pdbs_to_use_in_train,
            max_pdbs_to_use_in_val=args.max_pdbs_to_use_in_val,
            max_pdbs_to_use_in_test=args.max_pdbs_to_use_in_test,
            prevent_smiles_overlap=False  # DEBUG
        )

        # You'll always need the test data.
        test_data = self._get_data_from_split(
            args, moad, test, voxel_params, device, shuffle=False
        )

        all_pairs = []
        cnt = 0
        for batch in tqdm(test_data):
            if cnt > 200:
                break
            for b in batch[2]:
                all_pairs.append([b.receptor_name, b.fragment_smiles])
                # print(b)
            cnt = cnt + 1
        all_pairs.sort()
        import json
        import time
        with open("/mnt/extra/tmptmp." + str(int(time.time())) + ".txt", "w") as f:
            # f.write(json.dumps(all_pairs, indent=2))
            f.write(json.dumps(all_pairs).replace('["Receptor ', "\n").replace('", "', "\t").replace('"],', ""))
            # f.write("\n".join([a[0] for a in all_pairs]))
        print(len(all_pairs))
        import pdb; pdb.set_trace()

        trainer = self._init_trainer(args)

        ckpts = [c.strip() for c in ckpt.split(",")]       
        all_test_data = {
            "checkpoints": [{"name": c, "order": i + 1} for i, c in enumerate(ckpts)],
            "entries": []
        }
        for ckpt_idx, ckpt in enumerate(ckpts):
            model = self._init_model(args, ckpt)
            model.eval()

            predictions_per_rot = ensemble_helper.AveragedEnsembled(
                trainer, model, test_data, args.inference_rotations, device, ckpt
            )

            if ckpt_idx == 0:
                # Get the label set to use. Note that it only does this once
                # (for the first-checkpoint model), but I need model so I'm
                # leaving it in the loop.
                label_set_fingerprints, label_set_entry_infos = self._create_label_set_tensor(
                    args, device, 
                    existing_label_set_fps=predictions_per_rot.model.prediction_targets,
                    existing_label_set_entry_infos=predictions_per_rot.model.prediction_targets_entry_infos,
                    skip_test_set=True,
                    train=train, val=val, moad=moad, voxel_params=voxel_params,
                    lbl_set_codes=lbl_set_codes
                )

                # Get a PCA (or other) space defined by the label-set fingerprints.
                vis_rep_space = make_vis_rep_space_from_label_set_fingerprints(
                    label_set_fingerprints, 2
                )

                all_test_data["pcaPercentVarExplainedByEachComponent"] = [
                    100 * r 
                    for r in vis_rep_space.vis_rep.explained_variance_ratio_.tolist()
                ]

                avg_over_ckpts_of_avgs = torch.zeros(model.prediction_targets.shape, device=device)

            predictions_per_rot.finish(vis_rep_space)
            model, predictions_ensembled = predictions_per_rot.unpack()
            torch.add(avg_over_ckpts_of_avgs, predictions_ensembled, out=avg_over_ckpts_of_avgs)

            if ckpt_idx == 0:
                for i in range(predictions_per_rot.predictions_ensembled.shape[0]):
                    # Add in correct answers for all entries
                    correct_answer = predictions_per_rot.get_correct_answer_info(i)
                    all_test_data["entries"].append({
                        "correct": correct_answer,
                        "perCheckpoint": []
                    })

            # Calculate top_k metric for this checkpoint
            top_k_results = top_k(
                predictions_ensembled, model.prediction_targets, label_set_fingerprints,
                k=[1,8,16,32,64]
            )
            all_test_data["checkpoints"][ckpt_idx]["topK"] = {
                f'testTop{k}': float(top_k_results[k])
                for k in top_k_results
            }

            # Add info about the per-rotation predictions
            for entry_idx in range(len(predictions_ensembled)):
                entry = predictions_per_rot.get_predictions_info(entry_idx)
                all_test_data["entries"][entry_idx]["perCheckpoint"].append(entry)

            # Find most similar matches
            most_similar = most_similar_matches(
                predictions_ensembled, label_set_fingerprints, label_set_entry_infos, 
                NUM_MOST_SIMILAR_PER_ENTRY, vis_rep_space
            )
            for entry_idx in range(len(predictions_ensembled)):
                # Add closest compounds from label set.
                for predicted_entry_info, dist, viz_rep in most_similar[entry_idx]:
                    all_test_data["entries"][entry_idx]["perCheckpoint"][-1]["averagedPrediction"]["closestFromLabelSet"].append({
                        "smiles": predicted_entry_info.fragment_smiles,
                        "cosineDistToAveraged": dist,
                        "vizRepProjection": viz_rep[0]
                    })

        # Get the average of averages (across all checkpoints)
        torch.div(
            avg_over_ckpts_of_avgs, 
            torch.tensor(len(ckpts), device=device),
            out=avg_over_ckpts_of_avgs
        )

        # Calculate top-k metric of that average of averages
        top_k_results = top_k(
            avg_over_ckpts_of_avgs, model.prediction_targets, label_set_fingerprints,
            k=[1,8,16,32,64]
        )
        all_test_data["checkpoints"].append({
            "name": "Using average fingerprint over all checkpoints",
            "topK": {
                f'testTop{k}': float(top_k_results[k])
                for k in top_k_results
            }
        })

        # Get the fingerprints of the average of average outputs.
        avg_over_ckpts_of_avgs_viz = vis_rep_space.project(avg_over_ckpts_of_avgs)

        # For average of averages, Find most similar matches
        most_similar = most_similar_matches(
            avg_over_ckpts_of_avgs, label_set_fingerprints, label_set_entry_infos, 
            NUM_MOST_SIMILAR_PER_ENTRY, vis_rep_space
        )
        for entry_idx in range(len(avg_over_ckpts_of_avgs)):
            # Add closest compounds from label set.
            all_test_data["entries"][entry_idx]["avgOfCheckpoints"] = {
                "vizRepProjection": avg_over_ckpts_of_avgs_viz[entry_idx],
                "closestFromLabelSet": []
            }
            for predicted_entry_info, dist, viz_rep in most_similar[entry_idx]:
                all_test_data["entries"][entry_idx]["avgOfCheckpoints"]["closestFromLabelSet"].append({
                    "smiles": predicted_entry_info.fragment_smiles,
                    "cosineDistToAveraged": dist,
                    "vizRepProjection": viz_rep[0]
                })

        # import pdb; pdb.set_trace()

        jsn = json.dumps(all_test_data, indent=4)
        jsn = re.sub(r"([\-0-9\.]+?,)\n +?([\-0-9\.])", r"\1 \2", jsn, 0, re.MULTILINE)
        jsn = re.sub(r"\[\n +?([\-0-9\.]+?), ([\-0-9\.,]+?)\n +?\]", r"[\1, \2]", jsn, 0, re.MULTILINE)
        jsn = re.sub(r"\"Receptor ", '"', jsn, 0, re.MULTILINE)
        jsn = re.sub(r"\n +?\"dist", " \"dist", jsn, 0, re.MULTILINE)
        
        open("/mnt/extra/tmptmp.json", "w").write(jsn)

        # txt = ""
        # for entry in all_test_data["entries"]:
        #     txt += "Correct\n"
        #     txt += "\t".join([str(e) for e in entry["correct"]["vizRepProjection"]]) + "\t" + entry["correct"]["fragmentSmiles"] + "\n"
        #     txt += "averagedPrediction\n"
        #     txt += "\t".join([str(e) for e in entry["averagedPrediction"]["vizRepProjection"]]) + "\n"
        #     txt += "closestFromLabelSet\n"
        #     for close in entry["averagedPrediction"]["closestFromLabelSet"]:
        #         txt += "\t".join([str(e) for e in close["vizRepProjection"]]) + "\t" + close["smiles"] + "\n"
        #     txt += "predictionsPerRotation\n"
        #     for pred_per_rot in entry["predictionsPerRotation"]:
        #         txt += "\t".join([str(e) for e in pred_per_rot]) + "\n"
        #     txt = txt + "\n"

        # open("/mnt/extra/tmptmp.txt", "w").write(txt)

        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        with open('/mnt/extra/cProfile.txt', 'w+') as f:
            f.write(s.getvalue())


        import pdb; pdb.set_trace()

