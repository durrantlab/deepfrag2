import argparse
from functools import partial
from multiprocessing import Value
from typing import Any, Type, TypeVar, List, Optional, Tuple
from collagen.core.loader import DataLambda
from collagen.core.mol import Mol, mols_from_smi_file
from collagen.external.moad.types import MOAD_split
from tqdm.std import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
import torch

from ..checkpoints import MyModelCheckpoint, get_last_checkpoint
from .. import VoxelParams, VoxelParamsDefault, MultiLoader
from ..external import MOADInterface

from collagen.metrics import most_similar_matches, project_predictions_onto_label_set_pca_space, top_k

import re
import json


ENTRY_T = TypeVar("ENTRY_T")
TMP_T = TypeVar("TMP_T")
OUT_T = TypeVar("OUT_T")


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
            "--num_dataloader_workers",
            default=1,
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

    def _run_train(self, args: argparse.Namespace, ckpt: Optional[str]):
        trainer = self._init_trainer(args)
        model = self._init_model(args, ckpt)
        voxel_params = self._init_voxel_params(args)
        device = self._init_device(args)

        moad = MOADInterface(metadata=args.csv, structures=args.data)
        train, val, _ = moad.compute_split(
            args.split_seed, save_splits=args.save_splits,
            load_splits=args.load_splits
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
        existing_label_set_smis: List[str]=None,
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
                existing_label_set_fps, existing_label_set_smis, device=device
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

        if not ckpt:
            raise ValueError("Must specify a checkpoint in test mode")

        lbl_set_codes = [p.strip() for p in args.inference_label_sets.split(",")]

        if not "test" in lbl_set_codes:
            raise ValueError("To run in test mode, you must include the `test` label set")

        trainer = self._init_trainer(args)
        model = self._init_model(args, ckpt)
        voxel_params = self._init_voxel_params(args)
        device = self._init_device(args)

        moad = MOADInterface(metadata=args.csv, structures=args.data)
        train, val, test = moad.compute_split(
            args.split_seed, save_splits=args.save_splits,
            load_splits=args.load_splits
        )

        # You'll always need the test data.
        test_data = self._get_data_from_split(
            args, moad, test, voxel_params, device, shuffle=False
        )

        model.eval()

        # Intentionally keeping all predictions in a list instead of averaging
        # as you go. To allow for examining each rotations prediction. Must use
        # list (not zero tensor) because I don't think I can know the number of
        # items in test_data beforehand. TODO: Better way to do this?
        all_predictions_lst = []
        for i in range(args.inference_rotations):
            print(f"Inference rotation {i+1}/{args.inference_rotations}")
            trainer.test(model, test_data, verbose=True)
            all_predictions_lst.append(model.predictions)
        
        # Convert the list to a tensor now that you know what the dimensions
        # must be.
        all_predictions_tnsr = torch.zeros(
            (
                args.inference_rotations, 
                all_predictions_lst[0].shape[0], 
                all_predictions_lst[0].shape[1]
            ),
            device=device
        )
        for i in range(args.inference_rotations):
            all_predictions_tnsr[i] = all_predictions_lst[i]

        # Calculate the average predictions
        predictions_averaged = torch.sum(all_predictions_tnsr, dim=0)
        torch.div(
            predictions_averaged, 
            torch.tensor(args.inference_rotations, device=device),
            out=predictions_averaged
        )

        # predictions = torch.zeros(all_predictions_lst[0].shape, device=device)
        # for prediction in all_predictions_lst:
        #     torch.add(predictions, prediction, out=predictions)
        # torch.div(
        #     predictions, 
        #     torch.tensor(args.inference_rotations, device=device),
        #     out=predictions
        # )

        # Get the label set to use
        label_set_fingerprints, label_set_smis = self._create_label_set_tensor(
            args, device, 
            existing_label_set_fps=model.prediction_targets,
            existing_label_set_smis=model.prediction_targets_smis,
            skip_test_set=True,
            train=train, val=val, moad=moad, voxel_params=voxel_params,
            lbl_set_codes=lbl_set_codes
        )

        # avg_predictions = model.predictions
        num_predictions_per_entry = 5
        all_test_data = {
            "top_k": {},
            "entries": [
                {
                    "correct": {}, 
                    "prediction": {
                        "closests": [{} for __ in range(num_predictions_per_entry)]
                    }
                }
                for _ in range(predictions_averaged.shape[0])
            ]
        }

        # Calculate top_k metric
        top = top_k(
            predictions_averaged, model.prediction_targets, label_set_fingerprints,
            k=[1,8,16,32,64]
        )
        for k in top:
            all_test_data["top_k"][f'test_top_{k}'] = float(top[k])
            print(f'test_top_{k}',top[k])

        # Find most similar matches
        most_similar_idxs, most_similar_dists = most_similar_matches(predictions_averaged, label_set_fingerprints, 5)
        for entry_idx in range(most_similar_idxs.shape[0]):
            correct_smi = model.prediction_targets_smis[entry_idx]
            all_test_data["entries"][entry_idx]["correct"]["smiles"] = correct_smi
            
            dists = most_similar_dists[entry_idx]
            idxs = most_similar_idxs[entry_idx]
            smis = [label_set_smis[idx] for idx in idxs]
            
            hits = []
            for i, (d, s) in enumerate(zip(dists, smis)):
                all_test_data["entries"][entry_idx]["prediction"]["closests"][i]["smiles"] = s
                all_test_data["entries"][entry_idx]["prediction"]["closests"][i]["dist"] = float(d)
                hits.append("    " + s + " : " + str(float(d)))

            print("Correct answer: " + correct_smi)
            print("\n".join(hits))
            print("")

        all_rotations_onto_pca, correct_predicton_targets_onto_pca = project_predictions_onto_label_set_pca_space(
            all_predictions_tnsr, label_set_fingerprints, 2,
            correct_predicton_targets=model.prediction_targets
        )
        # To print out: https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib

        for entry_idx, rotations_onto_pcas in enumerate(all_rotations_onto_pca):
            all_test_data["entries"][entry_idx]["prediction"]["pca_per_rotations"] = [
                r.tolist() for r in rotations_onto_pcas
            ]
            all_test_data["entries"][entry_idx]["correct"]["pca_per_rotations"] = [
                r.tolist() for r in correct_predicton_targets_onto_pca[entry_idx]
            ]

        jsn = json.dumps(all_test_data, indent=4)
        jsn = re.sub(r"([\-0-9\.]+?,)\n .+?([\-0-9\.])", r"\1 \2", jsn, 0, re.MULTILINE)
        jsn = re.sub(r"\[\n +?([\-0-9\.]+?), ([\-0-9\.,]+?)\n +?\]", r"[\1, \2]", jsn, 0, re.MULTILINE)
        jsn = re.sub(r"\n +?\"dist", " \"dist", jsn, 0, re.MULTILINE)
        

        import pdb; pdb.set_trace()

