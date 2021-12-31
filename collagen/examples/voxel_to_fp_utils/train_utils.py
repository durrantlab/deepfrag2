import argparse
import time
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from collagen.checkpoints import MyModelCheckpoint, get_last_checkpoint
from collagen.core.loader import MultiLoader
from collagen.core.voxelizer import VoxelParamsDefault
import pytorch_lightning as pl

# JDD NO: torch.multiprocessing.set_sharing_strategy("file_system")

FP_SIZE = 2048


def disable_warnings():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    import prody

    prody.confProDy(verbosity="none")


def run_voxel_to_fp_model(
    args, ModelCls, InterfaceCls, DatasetCls, PreVoxelizeCls, BatchVoxelizeCls
):
    disable_warnings()

    vp = VoxelParamsDefault.DeepFrag

    model = ModelCls(voxel_features=vp.atom_featurizer.size() * 2, fp_size=FP_SIZE)

    # TODO: Code like below might be useful at inference time...
    #     print("Loading checkpoint from " + ckpt + "...")
    #     model = DeepFragModel.load_from_checkpoint(
    #         ckpt, voxel_features=vp.atom_featurizer.size() * 2, fp_size=FP_SIZE
    #     )

    interface = InterfaceCls(args.csv, args.data)
    train, val, test = interface.compute_split(seed=args.split_seed)

    train_frags = DatasetCls(
        interface, cache_file=args.cache, split=train, transform=PreVoxelizeCls(vp)
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
        .map(BatchVoxelizeCls(vp, args.cpu))
    )

    # Use below to debug errors in file loading and grid generation.
    # print(len(train_data))
    # import pdb; pdb.set_trace()
    # for t in train_data:
    #     dir(t)
    #     print("h")

    val_frags = DatasetCls(
        interface, cache_file=args.cache, split=val, transform=PreVoxelizeCls(vp)
    )
    val_data = (
        MultiLoader(
            val_frags,
            batch_size=1,  # Number of examples pulled at a time for loading data.
            shuffle=True,
            num_dataloader_workers=args.num_dataloader_workers,
            max_voxels_in_memory=args.max_voxels_in_memory,
        )
        # This is batch size from ML perspective. TODO: Check if hard coded
        # anywhere else.
        .batch(16).map(BatchVoxelizeCls(vp, args.cpu))
    )

    logger = None
    if args.wandb_project:
        logger = WandbLogger(project=args.wandb_project)
    else:
        logger = CSVLogger("logs", name="my_exp_name", flush_logs_every_n_steps=25)

    print(args)
    input("Anything to indicate batches? Didn't see anything. I think it's hard coded.")

    trainer = pl.Trainer.from_argparse_args(
        args,
        # default_root_dir="./.save",
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
    #     import pdb; pdb.set_trace()

    ckpt = get_last_checkpoint(args)
    if ckpt is not None:
        print("\n")
        print(
            "WARNING: Restarting training from where it previously left off ("
            + ckpt
            + ")."
        )
        print("\n")
        time.sleep(5)

    # trainer.tune(model)

    # May possibly speed things up:
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/36
    # Commented out because improvement was minor, if any, in quick benchmark.
    # torch.backends.cudnn.benchmark = True

    trainer.fit(model, train_data, val_data, ckpt_path=ckpt)


def add_args_voxel_to_fp_model():
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
    return args
