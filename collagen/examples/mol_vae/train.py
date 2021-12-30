import argparse

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from collagen.external.zinc import ZINCDatasetH5

from .model import MolVAE


def run(args):
    zinc = ZINCDatasetH5(args.data, make_3D=False)
    train_data = DataLoader(zinc, batch_size=8, shuffle=True, collate_fn=lambda x: x)

    model = MolVAE(
        atom_dim=13,
        bond_dim=3,
        z_size=64,
        z_select=256,
        z_bond=256,
        z_atom=256,
        dist_r=50,
        enc_steps=7,
        dec_steps=7,
        use_argmax=False,
        image_freq=50,
        kl_warm=100,
        kl_ramp=100,
        kl_hold=50,
        kl_max=0.05,
    )

    logger = None
    if args.wandb_project:
        logger = WandbLogger(project=args.wandb_project)
        logger.watch(model)

    trainer = pl.Trainer.from_argparse_args(
        args, default_root_dir="./.save", logger=logger
    )
    trainer.fit(model, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to zinc.h5")
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--wandb_project", required=False, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    run(args)
