"""The MOAD voxel model for training."""

from argparse import Namespace
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
from collagen.core.loader import DataLambda
from collagen.external.common.parent_interface import ParentInterface
from collagen.external.paired_csv.interface import PairedCsvInterface
from collagen.external.pdb_sdf_dir.interface import PdbSdfDirInterface
from collagen.model_parents.moad_voxel.inits import VoxelModelInits
from torchinfo import summary  # type: ignore
from collagen.external.moad.interface import MOADInterface
from collagen.external.common.split import create_train_val_test_splits

if TYPE_CHECKING:
    from collagen.model_parents.moad_voxel.moad_voxel import VoxelModelParent


class VoxelModelTrain(object):
    """A model for training on the MOAD dataset."""

    def __init__(self, parent: "VoxelModelParent"):
        """Initialize the class.

        Args:
            parent (VoxelModelParent): The parent class.
        """
        self.parent = parent

    def run_train(self, args: Namespace, ckpt_filename: Optional[str]):
        """Run training.

        Args:
            args (Namespace): The arguments passed to the program.
            ckpt_filename (Optional[str]): The checkpoint filename to use.
        """
        # Runs training.
        trainer = VoxelModelInits.init_trainer(args)
        data_interface, train_data, val_data = self.get_train_val_sets(args, False)

        # Below is helpful for debugging
        # for batch in train_data:
        #     receptors = [e.receptor_name.replace("Receptor ", "") for e in batch[2]]
        #     print(receptors)
        #     # print(batch)
        #     # import pdb; pdb.set_trace()
        #     continue

        model = self.parent.inits.init_model(args, ckpt_filename)

        model_stats = summary(model, (16, 10, 24, 24, 24), verbose=0)
        summary_str = str(model_stats)
        print(summary_str)

        trainer.fit(model, train_data, val_data, ckpt_path=ckpt_filename)

        self.parent.save_examples_used(model, args)

    def run_warm_starting(self, args: Namespace):
        """Run warm starting.

        Args:
            args (Namespace): The arguments passed to the program.
        """
        trainer = self.parent.inits.init_trainer(args)
        data_interface, train_data, val_data = self.get_train_val_sets(args, True)

        model = self.parent.inits.init_warm_model(args, data_interface)

        model_stats = summary(model, (16, 10, 24, 24, 24), verbose=0)
        summary_str = str(model_stats)
        print(summary_str)

        trainer.fit(model, train_data, val_data)

        self.parent.save_examples_used(model, args)

    def get_train_val_sets(
        self, args: Namespace, finetuning: bool
    ) -> Tuple[ParentInterface, DataLambda, Union[DataLambda, None],]:
        # NOTE: All interfaces should inherit from a base class.
        """Get the training and validation sets.

        Args:
            args (Namespace): The arguments passed to the program.
            finetuning (bool): Whether to fine-tune or train.

        Returns:
            Tuple[Any, DataLambda, DataLambda]: The MOAD, training and
                validation sets.
        """
        if args.custom_test_set_dir:
            raise Exception("The custom test set can only be used in inference mode")

        voxel_params = self.parent.inits.init_voxel_params(args)
        device = self.parent.inits.init_device(args)

        if not finetuning:
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

            # Training on BidingMOAD database
            if args.mode == "train":
                data_interface = MOADInterface(
                    metadata=args.every_csv,
                    structures_path=args.data_dir,
                    cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                    grid_width=voxel_params.width,
                    grid_resolution=voxel_params.resolution,
                    noh=args.noh,
                    discard_distant_atoms=args.discard_distant_atoms,
                )
            # Training on PDB/SDF files
            elif args.mode == "train_on_pdb_sdf_files":
                data_interface = PdbSdfDirInterface(
                    metadata=args.every_csv,
                    structures_path=args.data_dir,
                    cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                    grid_width=voxel_params.width,
                    grid_resolution=voxel_params.resolution,
                    noh=args.noh,
                    discard_distant_atoms=args.discard_distant_atoms,
                )
            else:
                raise ValueError(
                    "The training modes are 'train' for Biding MOAD Database, or 'train_on_pdb_sdf_files' for input PDB "
                    "and SDF files."
                )
        else:
            if (args.every_csv and not args.data_dir) or (args.data_dir and not args.every_csv) or \
                    (args.paired_data_csv and (args.every_csv or args.data_dir)):
                raise ValueError(
                    "For fine-tuning, the '--every_csv' and '--data_dir' parameters are specified when the input are "
                    "PDB/SDF files that are read from a CSV file, whereas the '--paired_data_csv' parameter is "
                    "specified when using paired data."
                )

            # Fine-tuning mode using a non-paired database other than MOAD
            if args.every_csv and args.data_dir:
                data_interface = PdbSdfDirInterface(
                    metadata=args.every_csv,
                    structures_path=args.data_dir,
                    cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                    grid_width=voxel_params.width,
                    grid_resolution=voxel_params.resolution,
                    noh=args.noh,
                    discard_distant_atoms=args.discard_distant_atoms,
                )
            # Fine-tuning mode using a paired database other than MOAD
            elif args.paired_data_csv:
                data_interface = PairedCsvInterface(
                    structures=args.paired_data_csv,
                    cache_pdbs_to_disk=args.cache_pdbs_to_disk,
                    grid_width=voxel_params.width,
                    grid_resolution=voxel_params.resolution,
                    noh=args.noh,
                    discard_distant_atoms=args.discard_distant_atoms,
                    use_prevalence=args.use_prevalence,
                )

        train_split, val_split, _ = create_train_val_test_splits(
            data_interface,
            seed=args.split_seed,
            fraction_train=args.fraction_train,
            fraction_val=args.fraction_val,
            prevent_smiles_overlap=True,
            save_splits=args.save_splits,
            load_splits=args.load_splits,
            max_pdbs_train=args.max_pdbs_train,
            max_pdbs_val=args.max_pdbs_val,
            max_pdbs_test=args.max_pdbs_test,
            split_method=args.split_method,
            butina_cluster_cutoff=args.butina_cluster_cutoff,
        )

        # pr = cProfile.Profile()
        # pr.enable()

        # pr.disable()
        # s = StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        # ps.print_stats()
        # open('cProfilez.txt', 'w+').write(s.getvalue())

        train_data: DataLambda = self.parent.utils.get_data_from_split(
            cache_file=args.cache,
            args=args,
            data_interface=data_interface,
            split=train_split,
            voxel_params=voxel_params,
            device=device,
        )
        print(f"Number of batches for the training data: {len(train_data)}")

        if len(val_split.targets) > 0:
            val_data: Union[DataLambda, None] = self.parent.utils.get_data_from_split(
                cache_file=args.cache,
                args=args,
                data_interface=data_interface,
                split=val_split,
                voxel_params=voxel_params,
                device=device,
            )
            print(f"Number of batches for the validation data: {len(val_data)}")
        else:
            val_data = None

        return data_interface, train_data, val_data
