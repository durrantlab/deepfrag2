import argparse
from pathlib import Path
from typing import List, Tuple

import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from collagen import (
    VoxelParamsDefault,
    AtomicNumFeaturizer,
    Mol,
    DelayedMolVoxel,
    VoxelParams,
    MultiLoader,
)
from collagen.external import MOADInterface, MOADPocketDataset
from collagen.util import rand_rot

from model import BindingSiteModel


def disable_warnings():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    import prody

    prody.confProDy(verbosity="none")


class PreVoxelize(object):
    """
    Pre-voxelize transform. Given a (receptor, parent, fragment) tuple, prepare to voxelize the receptor and parent
    and compute the fingerprint for the fragment.
    """

    def __init__(self, voxel_params: VoxelParams):
        self.voxel_params = voxel_params

    def __call__(
        self, rec: Mol, pos: "numpy.ndarray", neg: "numpy.ndarray"
    ) -> Tuple[DelayedMolVoxel, DelayedMolVoxel]:
        rot = rand_rot()
        return (
            rec.voxelize_delayed(self.voxel_params, center=pos, rot=rot),
            rec.voxelize_delayed(self.voxel_params, center=neg, rot=rot),
        )


class BatchVoxelize(object):
    """
    Voxelize multiple samples on the GPU.
    """

    def __init__(self, voxel_params: VoxelParams, cpu: bool):
        self.voxel_params = voxel_params
        self.cpu = cpu
        self.device = torch.device("cpu") if cpu else torch.device("cuda")

    def __call__(
        self, data: List[Tuple[DelayedMolVoxel, DelayedMolVoxel]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = [x for x in data if x is not None]
        if len(data) == 0:
            return None

        voxels = torch.zeros(
            size=self.voxel_params.tensor_size(batch=len(data) * 2),
            device=self.device,
        )

        targets = torch.zeros(size=(len(data) * 2, 1), device=self.device)

        for i in range(len(data)):
            rec_pos, rec_neg = data[i]

            rec_pos.voxelize_into(
                voxels, batch_idx=(i * 2), layer_offset=0, cpu=self.cpu
            )
            rec_neg.voxelize_into(
                voxels, batch_idx=(i * 2) + 1, layer_offset=0, cpu=self.cpu
            )

            targets[(i * 2)] = 1
            targets[(i * 2) + 1] = 0

        return (voxels, targets)


def run(args):
    disable_warnings()

    vp = VoxelParamsDefault.DeepFrag
    model = BindingSiteModel(voxel_features=vp.atom_featurizer.size())

    moad_path = Path(args.moad)
    moad_csv = next(moad_path.glob("./**/every.csv"))
    moad_data = moad_path

    moad = MOADInterface(moad_csv, moad_data)
    train, val, _ = moad.compute_split(seed=args.split_seed)

    train_set = MOADPocketDataset(
        moad,
        split=train,
        transform=PreVoxelize(vp),
        padding=10,
    )
    train_data = (
        MultiLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_dataloader_workers=args.num_dataloader_workers,
        )
        .batch(32)
        .map(BatchVoxelize(vp, args.cpu))
    )

    val_set = MOADPocketDataset(
        moad,
        split=val,
        transform=PreVoxelize(vp),
        padding=10,
    )
    val_data = (
        MultiLoader(
            val_set,
            batch_size=1,
            shuffle=True,
            num_dataloader_workers=args.num_dataloader_workers,
        )
        .batch(32)
        .map(BatchVoxelize(vp, args.cpu))
    )

    logger = None
    if args.wandb_project:
        logger = WandbLogger(project=args.wandb_project)

    trainer = pl.Trainer.from_argparse_args(
        args, default_root_dir="./.save", logger=logger
    )
    trainer.fit(model, train_data, val_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moad", required=True, help="Path to MOAD data")
    parser.add_argument(
        "--split_seed",
        required=False,
        default=1,
        type=int,
        help="Seed for TRAIN/VAL/TEST split.",
    )
    parser.add_argument(
        "--num_dataloader_workers", default=1, type=int, help="Number of workers for DataLoader"
    )
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--wandb_project", required=False, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    run(args)
