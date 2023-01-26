from argparse import Namespace
from typing import Optional
from torchinfo import summary
from collagen.external.moad.interface import MOADInterface, PdbSdfDirInterface
from collagen.external.moad.split import compute_dataset_split, full_moad_split


class MoadVoxelModelTrain(object):

    def run_train(self: "MoadVoxelModelParent", args: Namespace, ckpt: Optional[str]):

        # Runs training.
        trainer = self.init_trainer(args)
        moad, train_data, val_data = self.get_moad_train_val_sets(args, True)

        # Below is helpful for debugging
        # for batch in train_data:
        #     receptors = [e.receptor_name.replace("Receptor ", "") for e in batch[2]]
        #     print(receptors)
        #     # print(batch)
        #     # import pdb; pdb.set_trace()
        #     continue

        model = self.init_model(args, ckpt)

        # TODO: model.device is "cpu". Is that right? Shouldn't it be "cuda"?

        model_stats = summary(model, (16, 10, 24, 24, 24), verbose=0)
        summary_str = str(model_stats)
        print(summary_str)

        trainer.fit(model, train_data, val_data, ckpt_path=ckpt)

        self.save_examples_used(model, args)

    def run_warm_starting(self, args):

        trainer = self.init_trainer(args)
        moad, train_data, val_data = self.get_moad_train_val_sets(args, False)

        model = self.init_warm_model(args)

        model_stats = summary(model, (16, 10, 24, 24, 24), verbose=0)
        summary_str = str(model_stats)
        print(summary_str)

        trainer.fit(model, train_data, val_data)

        self.save_examples_used(model, args)

    def get_moad_train_val_sets(self, args, train: bool):

        if args.custom_test_set_dir:
            raise Exception("The custom test set can only be used in testing mode")

        voxel_params = self.init_voxel_params(args)
        device = self.init_device(args)

        if train:
            if args.every_csv is None:
                raise ValueError(
                    "For 'train' mode is required to specify the '--every_csv' parameter."
                )
            if args.butina_cluster_division:
                raise ValueError(
                    "Rational division based on Butina clustering is only for fine-tuning"
                )

            moad = MOADInterface(
                metadata=args.every_csv,
                structures_path=args.data_dir,
                cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                grid_width=voxel_params.width,
                grid_resolution=voxel_params.resolution,
                noh=args.noh,
                discard_distant_atoms=args.discard_distant_atoms,
            )
        else:
            if args.every_csv is not None:
                raise ValueError(
                    "For 'warm_starting' mode is not required to specify the '--every_csv' parameter."
                )

            moad = PdbSdfDirInterface(
                structures=args.data_dir,
                cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                grid_width=voxel_params.width,
                grid_resolution=voxel_params.resolution,
                noh=args.noh,
                discard_distant_atoms=args.discard_distant_atoms,
            )

        train, val, _ = compute_dataset_split(
            moad,
            seed=args.split_seed,
            fraction_train=args.fraction_train,
            fraction_val=args.fraction_val,
            save_splits=args.save_splits,
            load_splits=args.load_splits,
            max_pdbs_train=args.max_pdbs_train,
            max_pdbs_val=args.max_pdbs_val,
            max_pdbs_test=args.max_pdbs_test,
            butina_cluster_division=args.butina_cluster_division,
            butina_cluster_cutoff=args.butina_cluster_cutoff,
        )

        # pr = cProfile.Profile()
        # pr.enable()

        # pr.disable()
        # s = StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        # ps.print_stats()
        # open('cProfilez.txt', 'w+').write(s.getvalue())

        train_data = self.get_data_from_split(
            cache_file=args.cache, 
            args=args, 
            dataset=moad, 
            split=train, 
            voxel_params=voxel_params, 
            device=device
        )

        print("Number of batches for the training data: " + str(len(train_data)))
        if len(val.targets) > 0:
            val_data = self.get_data_from_split(
                cache_file=args.cache, 
                args=args, 
                dataset=moad, 
                split=val, 
                voxel_params=voxel_params, 
                device=device
            )
            print("Number of batches for the validation data: " + str(len(val_data)))
        else:
            val_data = None

        return moad, train_data, val_data
