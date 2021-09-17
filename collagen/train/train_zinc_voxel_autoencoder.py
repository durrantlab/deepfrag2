import argparse

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from collagen.data.zinc import ZINCDatasetH5
from collagen.data import VoxelParams, AtomicNumFeaturizer
from collagen.models.voxel_autoencoder import VoxelAutoencoder
from collagen.data import transforms


def run(args):
    vae = VoxelAutoencoder(latent_size=1024)

    vp = VoxelParams(
        resolution=0.75, width=24, atom_featurizer=AtomicNumFeaturizer([1, 6, 7, 8, 15])
    )

    zinc = ZINCDatasetH5(
        args.zinc_path, make_3D=True, transform=transforms.Voxelize(vp, cpu=args.cpu)
    )

    data = DataLoader(zinc, batch_size=16)

    logger = None
    if args.wandb_project:
        logger = WandbLogger(project=args.wandb_project)

    trainer = pl.Trainer.from_argparse_args(
        args, default_root_dir="./.save", logger=logger, limit_train_batches=500
    )
    trainer.fit(vae, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zinc_path", required=True, help="Path to zinc.h5")
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--wandb_project", required=False, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    run(args)
