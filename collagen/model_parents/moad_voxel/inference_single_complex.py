"""The inference mode for the MOAD voxel model."""

from argparse import Namespace
import cProfile
from io import StringIO
import torch  # type: ignore
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from collagen.core.molecules.mol import Mol
from collagen.metrics.metrics import most_similar_matches
import prody  # type: ignore
import numpy as np  # type: ignore
from collagen.util import rand_rot
import rdkit  # type: ignore
from rdkit import Chem  # type: ignore
import pytorch_lightning as pl  # type: ignore
import os
from collagen.model_parents.moad_voxel.inference import Inference

if TYPE_CHECKING:
    from collagen.model_parents.moad_voxel.moad_voxel import VoxelModelParent


class InferenceSingleComplex(Inference):
    """A model for inference."""

    def __init__(self, model_parent: Any):
        """Initialize the class.

        Args:
            parent (VoxelModelParent): The parent class.
        """
        Inference.__init__(self, model_parent)

        self.output_pred = None
        self.output_path = None

    def get_prediction(self):
        return self.output_pred

    def get_prediction_path(self):
        return self.output_path

    def _validate_run_test(self, args: Namespace, ckpt_filename: Optional[str]):
        """Validate the arguments for inference mode.

        Args:
            self: This object
            args (Namespace): The user arguments.
            ckpt (Optional[str]): The checkpoint to load.

        Raises:
            ValueError: If the arguments are invalid.
        """
        super()._validate_run_test(args, ckpt_filename)

        if args.receptor is None:
            raise ValueError("Must specify a receptor (--receptor) in inference mode")
        if args.ligand is None:
            raise ValueError("Must specify a ligand (--ligand) in inference mode")
        if args.branch_atm_loc_xyz is None:
            raise ValueError(
                "Must specify a center (--branch_atm_loc_xyz) in inference mode"
            )

    def run_test(self, args: Namespace, ckpt_filename: str):
        """Run a model on the test and evaluates the output.

        Args:
            self: This object
            args (Namespace): The user arguments.
            ckpt_filename (Optional[str]): The checkpoint to load.

        Returns:
            Union[Dict[str, Any], None]: The results dictionary, or None if
                save_results_to_disk is True.
        """
        self._validate_run_test(args, ckpt_filename)

        print(
            f"Using the operator {args.aggregation_rotations} to aggregate the inferences."
        )

        pr = cProfile.Profile()
        pr.enable()

        voxel_params = self.parent.inits.init_voxel_params(args)
        device = self.parent.inits.init_device(args)

        # Load the receptor
        with open(args.receptor, "r") as f:
            m = prody.parsePDBStream(StringIO(f.read()), model=1)
        prody_mol = m.select("all")
        recep = Mol.from_prody(prody_mol)
        center = np.array(
            [float(v.strip()) for v in args.branch_atm_loc_xyz.split(",")]
        )

        # Load the ligand
        suppl = Chem.SDMolSupplier(str(args.ligand))
        rdmol = [x for x in suppl if x is not None][0]
        lig = Mol.from_rdkit(rdmol, strict=False)

        ckpts = [c.strip() for c in ckpt_filename.split(",")]
        fps = []
        for ckpt_filename in ckpts:
            print(f"Using checkpoint {ckpt_filename}")
            model = self.parent.inits.init_model(args, ckpt_filename)
            model.eval()

            # You're iterating through multiple checkpoints. This allows output
            # from multiple trained models to be averaged.
            for r in range(args.rotations):
                print(f"    Rotation #{(r + 1)}")

                rot = rand_rot()
                recep_vox = recep.voxelize(
                    voxel_params, cpu=device, center=center, rot=rot
                )
                lig_vox = lig.voxelize(voxel_params, cpu=device, center=center, rot=rot)

                # Stack the receptor and ligand tensors
                num_features = recep_vox.shape[1] + lig_vox.shape[1]
                dim1 = lig_vox.shape[2]
                dim2 = lig_vox.shape[3]
                dim3 = lig_vox.shape[4]
                voxel = torch.cat([recep_vox[0], lig_vox[0]]).reshape(
                    [1, num_features, dim1, dim2, dim3]
                )
                fps.append(model.forward(voxel))

        avg_over_ckpts_of_avgs = torch.mean(torch.stack(fps), dim=0)

        # Now get the label sets to use.
        label_set_fingerprints, label_set_entry_infos = self._create_label_set(
            args,
            device,
            self._read_BindingMOAD_database(args, voxel_params) if args.every_csv and args.data_dir else None,
            voxel_params,
            existing_label_set_fps=torch.empty(0, dtype=torch.float32),
            existing_label_set_entry_infos=[],
            skip_test_set=True,
            train_split=None,
            val_split=None,
            lbl_set_codes=None,
        )

        most_similar = most_similar_matches(
            avg_over_ckpts_of_avgs,
            label_set_fingerprints,
            label_set_entry_infos,
            args.num_inference_predictions,
        )

        self.output_pred = {
            "most_similar": most_similar[0],
            "fps_per_rot": fps,
            "fps_avg": avg_over_ckpts_of_avgs,
        }

        self.output_pred["fps_per_rot"] = [
            v.detach().numpy().tolist() for v in self.output_pred["fps_per_rot"]
        ]
        self.output_pred["fps_avg"] = self.output_pred["fps_avg"].detach().numpy().tolist()

<<<<<<< HEAD
        self.output_path = args.default_root_dir + "predictions_Single_Complex" + os.sep + (os.path.basename(os.path.relpath(args.receptor)) + "_" + os.path.basename(os.path.relpath(args.ligand))) + ".results"
=======
        self.output_path = args.default_root_dir + os.sep + "predictions_Single_Complex" + os.sep + (os.path.basename(os.path.relpath(args.receptor)) + "_" + os.path.basename(os.path.relpath(args.ligand)))
>>>>>>> cdd221c4e2049e8167993ef1e3f43222ab2fae20
        os.makedirs(self.output_path, exist_ok=True)
        output_file = (
            self.output_path
            + os.sep
            + str(args.branch_atm_loc_xyz).replace(',', '_')
            + "_output.pt"
        )
        torch.save(
            self.output_pred,
            output_file,
        )

        # If you get here, you are saving the results to disk (default).
        print("")
        with open(f"{self.output_path}{os.sep}{str(args.branch_atm_loc_xyz).replace(',', '_')}_inference_out.tsv", "w") as f:
            header = "SMILES\tScore (Cosine Similarity)"
            f.write(f"{header}\n")
            print(header)
            for entry, score_cos_similarity, _ in most_similar[0]:
                line = f"{entry.fragment_smiles}\t{score_cos_similarity:.3f}"
                f.write(line + "\n")
                print(line)

        # TODO: Cesar: Need to check on some known answers as a "sanity check".

        # TODO: Cesar:  Can we add the fragments in most_similar[0] to the parent
        # molecule, to make a composite ligand ready for docking?
