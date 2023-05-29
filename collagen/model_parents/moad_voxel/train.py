"""The MOAD voxel model for training."""

from argparse import Namespace
from typing import Any, Optional, Tuple
from collagen.core.loader import DataLambda
from torchinfo import summary
from collagen.external.moad.interface import MOADInterface, PdbSdfDirInterface, PairedPdbSdfCsvInterface
from collagen.external.moad.split import compute_dataset_split


class MoadVoxelModelTrain(object):

    """A model for training on the MOAD dataset."""

    def run_train(
        self: "MoadVoxelModelParent", args: Namespace, ckpt_filename: Optional[str]
    ):
        """Run training.
        
        Args:
            args (Namespace): The arguments passed to the program.
            ckpt_filename (Optional[str]): The checkpoint filename to use.
        """
        # Runs training.
        trainer = self.init_trainer(args)
        moad, train_data, val_data = self.get_train_val_sets(args, True)

        # Below is helpful for debugging
        # for batch in train_data:
        #     receptors = [e.receptor_name.replace("Receptor ", "") for e in batch[2]]
        #     print(receptors)
        #     # print(batch)
        #     # import pdb; pdb.set_trace()
        #     continue

        model = self.init_model(args, ckpt_filename)

        # TODO: model.device is "cpu". Is that right? Shouldn't it be "cuda"?

        model_stats = summary(model, (16, 10, 24, 24, 24), verbose=0)
        summary_str = str(model_stats)
        print(summary_str)

        trainer.fit(model, train_data, val_data, ckpt_path=ckpt_filename)

        self.save_examples_used(model, args)

    def run_warm_starting(self, args: Namespace):
        """Run warm starting.
        
        Args:
            args (Namespace): The arguments passed to the program.
        """
        trainer = self.init_trainer(args)
        moad, train_data, val_data = self.get_train_val_sets(args, False)

        model = self.init_warm_model(args, moad)

        model_stats = summary(model, (16, 10, 24, 24, 24), verbose=0)
        summary_str = str(model_stats)
        print(summary_str)

        trainer.fit(model, train_data, val_data)

        self.save_examples_used(model, args)

    def get_train_val_sets(
        self, args: Namespace, train: bool
    ) -> Tuple[Any, DataLambda, DataLambda]:
        """Get the training and validation sets.

        Args:
            args (Namespace): The arguments passed to the program.
            train (bool): Whether to train or fine-tune.

        Returns:
            Tuple[Any, DataLambda, DataLambda]: The MOAD, training and
                validation sets.
        """
        if args.custom_test_set_dir:
            raise Exception("The custom test set can only be used in inference mode")

        voxel_params = self.init_voxel_params(args)
        device = self.init_device(args)

        if train:
            if args.paired_data_csv:
                raise ValueError(
                    "For 'train' mode, you must not specify the '--paired_data_csv' parameter."
                )
            if not args.data_dir:
                raise ValueError(
                    "For 'train' mode, you must specify the '--data_dir' parameter."
                )
            if not args.every_csv:
                raise ValueError(
                    "For 'train' mode, you must specify the '--every_csv' parameter."
                )
            if args.butina_cluster_cutoff:
                raise ValueError(
                    "Rational division based on Butina clustering is only for fine-tuning."
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
            if args.every_csv:
                raise ValueError(
                    "For 'fine-tuning' mode, you must not specify the '--every_csv' parameter."
                )
            if (args.data_dir and args.paired_data_csv) or (not args.data_dir and not args.paired_data_csv):
                raise ValueError(
                    "For 'fine-tuning' mode, you must specify the '--data_dir' parameter or the '--paired_data_csv' parameter."
                )

            if args.data_dir:  # for fine-tuning mode using a non-paired database other than MOAD
                moad = PdbSdfDirInterface(
                    structures_dir=args.data_dir,
                    cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                    grid_width=voxel_params.width,
                    grid_resolution=voxel_params.resolution,
                    noh=args.noh,
                    discard_distant_atoms=args.discard_distant_atoms
                )
            else:  # for fine-tuning mode using a paired database other than MOAD
                moad = PairedPdbSdfCsvInterface(
                    structures=args.paired_data_csv,
                    cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                    grid_width=voxel_params.width,
                    grid_resolution=voxel_params.resolution,
                    noh=args.noh,
                    discard_distant_atoms=args.discard_distant_atoms
                )

        train, val, _ = compute_dataset_split(
            moad,
            seed=args.split_seed,
            fraction_train=args.fraction_train,
            fraction_val=args.fraction_val,
            prevent_smiles_overlap=True,
            save_splits=args.save_splits,
            load_splits=args.load_splits,
            max_pdbs_train=args.max_pdbs_train,
            max_pdbs_val=args.max_pdbs_val,
            max_pdbs_test=args.max_pdbs_test,
            butina_cluster_cutoff=args.butina_cluster_cutoff,
        )

        # pr = cProfile.Profile()
        # pr.enable()

        # pr.disable()
        # s = StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        # ps.print_stats()
        # open('cProfilez.txt', 'w+').write(s.getvalue())

        train_data: DataLambda = self.get_data_from_split(
            cache_file=args.cache,
            args=args,
            dataset=moad,
            split=train,
            voxel_params=voxel_params,
            device=device,
        )
        print(f"Number of batches for the training data: {len(train_data)}")

        if len(val.targets) > 0:
            val_data: DataLambda = self.get_data_from_split(
                cache_file=args.cache,
                args=args,
                dataset=moad,
                split=val,
                voxel_params=voxel_params,
                device=device,
            )
            print(f"Number of batches for the validation data: {len(val_data)}")
        else:
            val_data = None

        return moad, train_data, val_data
