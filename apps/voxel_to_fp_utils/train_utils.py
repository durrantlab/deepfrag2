import argparse
import time
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from tqdm.std import tqdm
from collagen.checkpoints import MyModelCheckpoint, get_last_checkpoint
from collagen.core.loader import MultiLoader
from collagen.core.voxelizer import VoxelParamsDefault
import pytorch_lightning as pl
import torch
from torch import nn


# JDD NO: torch.multiprocessing.set_sharing_strategy("file_system")

FP_SIZE = 2048

_cos = nn.CosineSimilarity(dim=0, eps=1e-6)

def disable_warnings():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    import prody

    prody.confProDy(verbosity="none")


def get_trainer(args):
    logger = None
    if args.wandb_project:
        logger = WandbLogger(project=args.wandb_project)
    else:
        logger = CSVLogger(
            "logs", name="my_exp_name", flush_logs_every_n_steps=args.log_every_n_steps
        )

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
        log_every_n_steps=args.log_every_n_steps,
        # fast_dev_run=True,
        # callbacks=[ModelSummary(max_depth=-1), DeviceStatsMonitor()],
        # overfit_batches=0.001,
        # track_grad_norm=2,
        # limit_train_batches=0.0001,
        # limit_val_batches=0.0001
    )

    return trainer


def inference_voxel_to_fp_model(
    args,
    ModelCls,
    InterfaceCls,
    DatasetCls,
    PreVoxelizeCls,
    BatchVoxelizeCls,
    voxel_params,
):
    ckpt = get_last_checkpoint(args)
    print("Loading checkpoint from " + ckpt + "...")

    model = ModelCls.load_from_checkpoint(
        ckpt, voxel_features=voxel_params.atom_featurizer.size() * 2, fp_size=FP_SIZE
    )

    # On cpu by default
    if not args.cpu:
        model.to("cuda:0")  # TODO: Should be hardcoded?

    interface = InterfaceCls(args.csv, args.data)
    train, val, test = interface.compute_split(seed=args.split_seed)

    test_dataset = DatasetCls(
        interface,
        cache_file=args.cache,
        split=test,
        transform=PreVoxelizeCls(voxel_params),
    )

    test_data = (
        MultiLoader(
            test_dataset,
            # Number of examples pulled at a time for loading data.
            batch_size=1,
            shuffle=True,
            num_dataloader_workers=args.num_dataloader_workers,
            max_voxels_in_memory=args.max_voxels_in_memory,
        )
        # This is batch size from ML perspective.
        .batch(args.batch_size).map(BatchVoxelizeCls(voxel_params, args.cpu))
    )

    all_preds = {}
    all_true_fps = {}

    num_rots = 2  # TODO: Should be user-specified parameter

    # See
    # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#inference-in-research

    model.eval()
    model.freeze()  # necessary?

    with torch.no_grad():
        for i in tqdm(range(num_rots), desc="Rotations..."):  # number random rotations
            for data in tqdm(test_data, desc="Running inference on test set..."):
                # Note that data could contain multiple fingerprints, the number
                # in args.batch_size.

                # TODO: Harrison, am I doing inference right here? I

                voxels, true_fps, recep_names, lig_names = data
                preds = model(voxels)
                fp_pairs = zip(preds, true_fps)
                names = zip(recep_names, lig_names)

                for ns, fp_pair in zip(names, fp_pairs):
                    recep_name, lig_name = ns
                    pred, true_fp = fp_pair

                    label = recep_name + "---" + lig_name
                    all_true_fps[label] = true_fp
                    
                    if not label in all_preds:
                        # all_preds[label] = p.cpu().numpy()
                        all_preds[label] = pred
                    else:
                        # all_preds[label] = all_preds[label] + p.cpu().numpy()
                        torch.add(all_preds[label], pred, out=all_preds[label])

        # tensor([1.6018, 0.5229, 0.5466,  ..., 0.9844, 1.0390, 1.5191], device='cuda:0')
        # array([1.6018307 , 0.52291   , 0.54655963, ..., 0.9843685 , 1.0389928 ,
        #     1.519117  ], dtype=float32)
        # Calculate the average fingerprints
        
        for label in all_preds:
            torch.mul(all_preds[label], 1.0 / num_rots, out=all_preds[label])
            # all_preds[label] = all_preds[label] / float(num_rots)
        
        # TODO: Here, I'm just using the test set as the label set. But you'll
        # want to make it so this can be customized too.
        keys = all_preds.keys()

        for k in keys:
            print(k)
            pred = all_preds[k]
            dists = []
            for k2, fp in all_true_fps.items():
                dissimilarity = float(1 - _cos(pred, fp))
                dists.append((dissimilarity, k2))
            
            dists.sort()
            for d in dists:
                print(d)
            print("=====")


        # For the purposes of testing
        # ****

        # all_preds["Receptor 1woq---BGC:B:291"]
        # tensor([0.4005, 0.1307, 0.1366,  ..., 0.2461, 0.2597, 0.3798], device='cuda:0')
        # array([0.40045768, 0.1307275 , 0.13663991, ..., 0.24609213, 0.2597482 ,
        # 0.37977925], dtype=float32)

        # array([0.290141  , 0.05857408, 0.08193754, ..., 0.18720812, 0.19347991,
        # 0.33487868], dtype=float32)

        # import pdb; pdb.set_trace()
        # print("H")

        #
        # pretrained_model.freeze()
        # y_hat = pretrained_model(x)


def train_voxel_to_fp_model(
    args, ModelCls, InterfaceCls, DatasetCls, PreVoxelizeCls, BatchVoxelizeCls
):
    disable_warnings()
    vp = VoxelParamsDefault.DeepFrag

    if args.mode == "test":
        # Run inference on test set.
        inference_voxel_to_fp_model(
            args,
            ModelCls,
            InterfaceCls,
            DatasetCls,
            PreVoxelizeCls,
            BatchVoxelizeCls,
            vp,
        )
        return

    trainer = get_trainer(args)

    model = ModelCls(voxel_features=vp.atom_featurizer.size() * 2, fp_size=FP_SIZE)

    interface = InterfaceCls(args.csv, args.data)
    train, val, test = interface.compute_split(seed=args.split_seed)

    train_dataset = DatasetCls(
        interface, cache_file=args.cache, split=train, transform=PreVoxelizeCls(vp)
    )
    train_data = (
        MultiLoader(
            train_dataset,
            # Number of examples pulled at a time for loading data.
            batch_size=1,
            shuffle=True,
            num_dataloader_workers=args.num_dataloader_workers,
            max_voxels_in_memory=args.max_voxels_in_memory,
        )
        # This is batch size from ML perspective.
        .batch(args.batch_size).map(BatchVoxelizeCls(vp, args.cpu))
    )

    # Use below to debug errors in file loading and grid generation.
    # print(len(train_data))
    # import pdb; pdb.set_trace()
    # for t in train_data:
    #     dir(t)
    #     print("h")

    val_dataset = DatasetCls(
        interface, cache_file=args.cache, split=val, transform=PreVoxelizeCls(vp)
    )
    val_data = (
        MultiLoader(
            val_dataset,
            # Number of examples pulled at a time for loading data.
            batch_size=1,
            shuffle=True,
            num_dataloader_workers=args.num_dataloader_workers,
            max_voxels_in_memory=args.max_voxels_in_memory,
        )
        # This is batch size from ML perspective.
        .batch(args.batch_size).map(BatchVoxelizeCls(vp, args.cpu))
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
        help="The size of the batch",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Can be train or test. If train, trains the model. If test, runs inference on the test set. Defaults to train.",
        default="train",
    )

    # parser.add_argument(
    #     "--log_every_n_steps",
    #     type=int,
    #     required=False,
    #     default=25,
    #     help="How often to log data",
    # )

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # JDD added to see if helps with error:
    # https://stackoverflow.com/questions/67876741/unable-to-mmap-1024-bytes-cannot-allocate-memory-even-though-there-is-more-t/67969244#67969244
    # args.use_multiprocessing = False

    return args
