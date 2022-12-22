from argparse import Namespace
from collagen.external.moad.interface import MOADInterface
from collagen.external.moad.split import compute_dataset_split

# Code to identify the best learning rate for the model. I ended up not using
# it.


class MoadVoxelModelLRFinder(object):
    def run_lr_finder(self: "MoadVoxelModelParent", args: Namespace):
        # Update value of auto_lr_find
        args = vars(args)
        args["auto_lr_find"] = True
        args = Namespace(**args)

        trainer = self.init_trainer(args)
        model = self.init_model(args, None, None)
        voxel_params = self.init_voxel_params(args)
        device = self.init_device(args)

        moad = MOADInterface(
            metadata=args.every_csv,
            structures_path=args.data_dir,
            cache_pdbs_to_disk=args.cache_pdbs_to_disk,
            grid_width=voxel_params.width,
            grid_resolution=voxel_params.resolution,
            noh=args.noh,
            discard_distant_atoms=args.discard_distant_atoms,
        )
        train, val, _ = compute_dataset_split(
            moad,
            args.split_seed,
            save_splits=args.save_splits,
            load_splits=args.load_splits,
            max_pdbs_train=args.max_pdbs_train,
            max_pdbs_val=args.max_pdbs_val,
            max_pdbs_test=args.max_pdbs_test,
        )

        train_data = self.get_data_from_split(
            args, moad, train, voxel_params, device
        )

        val_data = self.get_data_from_split(
            args, moad, val, voxel_params, device
        )

        lr_finder = trainer.tuner.lr_find(model, train_data, val_data)
        print("Suggested learning rate:", lr_finder.suggestion())
