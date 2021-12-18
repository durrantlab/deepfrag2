import argparse
from typing import List, Tuple

import torch

torch.multiprocessing.set_sharing_strategy("file_system")

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger

from collagen import (
    VoxelParamsDefault,
    AtomicNumFeaturizer,
    Mol,
    DelayedMolVoxel,
    VoxelParams,
    MultiLoader,
)
from collagen.external import MOADInterface, MOADFragmentDataset
from collagen.util import rand_rot

from collagen.examples.deeplig.model import DeepFragModel

FP_SIZE = 2048


def disable_warnings():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    import prody

    prody.confProDy(verbosity="none")


class PreVoxelize(object):
    """
    Pre-voxelize transform. Given a (receptor, parent, fragment) tuple, prepare
    to voxelize the receptor and parent and compute the fingerprint for the
    fragment.
    """

    def __init__(self, voxel_params: VoxelParams):
        self.voxel_params = voxel_params

    def __call__(
        self, rec: Mol, parent: Mol, frag: Mol
    ) -> Tuple[DelayedMolVoxel, DelayedMolVoxel, torch.Tensor]:
        rot = rand_rot()
        return (
            rec.voxelize_delayed(self.voxel_params, center=frag.connectors[0], rot=rot),
            parent.voxelize_delayed(
                self.voxel_params, center=frag.connectors[0], rot=rot
            ),
            torch.tensor(frag.fingerprint("rdk10", 2048)),
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
        self, data: List[Tuple[DelayedMolVoxel, DelayedMolVoxel, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        voxels = torch.zeros(
            size=self.voxel_params.tensor_size(batch=len(data), feature_mult=2),
            device=self.device,
        )

        fingerprints = torch.zeros(size=(len(data), FP_SIZE), device=self.device)

        for i in range(len(data)):
            rec, parent, frag = data[i]

            rec.voxelize_into(voxels, batch_idx=i, layer_offset=0, cpu=self.cpu)

            parent.voxelize_into(
                voxels,
                batch_idx=i,
                layer_offset=self.voxel_params.atom_featurizer.size(),
                cpu=self.cpu,
            )

            fingerprints[i] = frag

        return (voxels, fingerprints)


def run(args):
    disable_warnings()

    vp = VoxelParamsDefault.DeepFrag

    model = DeepFragModel(voxel_features=vp.atom_featurizer.size() * 2, fp_size=FP_SIZE)

    moad = MOADInterface(args.csv, args.data)
    train, val, _ = moad.compute_split(seed=args.split_seed)

    train_frags = MOADFragmentDataset(
        moad, cache_file=args.cache, split=train, transform=PreVoxelize(vp)
    )
    train_data = (
        MultiLoader(
            train_frags,
            batch_size=1,
            shuffle=True,
            num_dataloader_workers=args.num_dataloader_workers,
            max_voxels_in_memory=args.max_voxels_in_memory,
        )
        .batch(16)
        .map(BatchVoxelize(vp, args.cpu))
    )

    val_frags = MOADFragmentDataset(
        moad, cache_file=args.cache, split=val, transform=PreVoxelize(vp)
    )
    val_data = (
        MultiLoader(
            val_frags,
            batch_size=1,
            shuffle=True,
            num_dataloader_workers=args.num_dataloader_workers,
            max_voxels_in_memory=args.max_voxels_in_memory,
        )
        .batch(16)
        .map(BatchVoxelize(vp, args.cpu))
    )

    logger = None
    if args.wandb_project:
        logger = WandbLogger(project=args.wandb_project)
    else:
        logger = CSVLogger("logs", name="my_exp_name", flush_logs_every_n_steps=25)

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir="./.save",
        logger=logger,
        # Below for debugging
        log_every_n_steps=25,
        # fast_dev_run=True,
        # callbacks=[ModelSummary(max_depth=-1), DeviceStatsMonitor()],
        # overfit_batches=0.001,
        # track_grad_norm=2,
        # limit_train_batches=0.0001,
        # limit_val_batches=0.0001
    )

    # Use below to debug errors in file loading and grid generation.
    # print(len(train_data))
    # for t in train_data:
    #     dir(t)
    #     print("h")
    # import pdb; pdb.set_trace()

    trainer.fit(model, train_data, val_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to MOAD every.csv")
    parser.add_argument(
        "--data", required=True, help="Path to MOAD root structure folder"
    )
    parser.add_argument("--cache", required=True, help="Path to MOAD cache.json file")
    parser.add_argument(
        "--split_seed",
        required=False,
        default=1,
        type=int,
        help="Seed for TRAIN/VAL/TEST split.",
    )
    parser.add_argument(
        "--num_dataloader_workers",
        default=1,
        type=int,
        help="Number of workers for DataLoader",
    )
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--wandb_project", required=False, default=None)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    run(args)
