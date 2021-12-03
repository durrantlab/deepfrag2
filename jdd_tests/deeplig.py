import sys
sys.path.append('..')

from collagen.examples.deeplig import train
import argparse
import pytorch_lightning as pl

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
    "--num_dataloader_workers", default=1, type=int, help="Number of workers for DataLoader"
)
parser.add_argument("--cpu", default=False, action="store_true")
parser.add_argument("--wandb_project", required=False, default=None)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

train.run(args)

