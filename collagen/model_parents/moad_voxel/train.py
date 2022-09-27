from argparse import Namespace
from typing import Optional
from torchinfo import summary
from collagen.external import MOADInterface
from collagen.external.moad.split import compute_moad_split


class MoadVoxelModelTrain(object):
    def run_train(self: "MoadVoxelModelParent", args: Namespace, ckpt: Optional[str]):
        # Runs training.

        trainer = self.init_trainer(args)
        voxel_params = self.init_voxel_params(args)
        device = self.init_device(args)

        moad = MOADInterface(
            metadata=args.csv,
            structures=args.data,
            cache_pdbs_to_disk=args.cache_pdbs_to_disk,
            grid_width=voxel_params.width,
            grid_resolution=voxel_params.resolution,
            noh=args.noh,
            discard_distant_atoms=args.discard_distant_atoms,
        )

        train, val, _ = compute_moad_split(
            moad,
            args.split_seed,
            save_splits=args.save_splits,
            load_splits=args.load_splits,
            max_pdbs_train=args.max_pdbs_train,
            max_pdbs_val=args.max_pdbs_val,
            max_pdbs_test=args.max_pdbs_test,
        )

        # pr = cProfile.Profile()
        # pr.enable()

        # pr.disable()
        # s = StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        # ps.print_stats()
        # open('cProfilez.txt', 'w+').write(s.getvalue())

        train_data = self.get_data_from_split(args, moad, train, voxel_params, device)
        val_data = self.get_data_from_split(args, moad, val, voxel_params, device)

        model = self.init_model(args, ckpt)

        model_stats = summary(model, (16, 10, 24, 24, 24), verbose=0)
        summary_str = str(model_stats)
        print(summary_str)

        trainer.fit(model, train_data, val_data, ckpt_path=ckpt)

        self._save_examples_used(model, args)
