"""The inference mode for the MOAD voxel model."""

from argparse import ArgumentParser, Namespace
import cProfile
from io import StringIO
import pstats
from collagen.model_parents.moad_voxel.test_inference_utils import (
    remove_redundant_fingerprints,
)
import torch
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from collagen.core.molecules.mol import Mol, mols_from_smi_file
from collagen.external.moad.types import Entry_info
from collagen.metrics.metrics import most_similar_matches
import prody
import numpy as np
from collagen.util import rand_rot
import rdkit
from rdkit import Chem
import pytorch_lightning as pl
import os
import json

if TYPE_CHECKING:
    from collagen.model_parents.moad_voxel.moad_voxel import MoadVoxelModelParent

class MoadVoxelModelInference(object):

    """A model for inference."""

    def create_inference_label_set(
        self: "MoadVoxelModelParent",
        args: Namespace,
        device: torch.device,
        smi_files: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:
        """Create a label set (look-up) tensor and smiles list for testing.
        Can be comprised of the fingerprints in the train and/or test and/or
        val sets, as well as SMILES strings from a file.

        Args:
            self (MoadVoxelModelParent): This object
            args (Namespace): The user arguments.
            device (torch.device): The device to use.
            smi_files (List[str]): The file(s) containing SMILES strings.

        Returns:
            Tuple[torch.Tensor, List[str]]: The updated fingerprint
                tensor and smiles list.
        """
        # Can we cache the label_set_fps and label_set_smis variables to a disk
        # to not have to recalculate them every time? It can be a pretty
        # expensive calculation.

        # Get fingerprints from SMI files.
        fp_tnsrs_from_smi_file = []
        label_set_smis = []
        for filename in smi_files:
            for smi, mol in mols_from_smi_file(filename):
                fp_tnsrs_from_smi_file.append(
                    torch.tensor(
                        mol.fingerprint(args.fragment_representation, args.fp_size),
                        dtype=torch.float32,
                        device=device,
                        requires_grad=False,
                    ).reshape((1, args.fp_size))
                )
                label_set_smis.append(smi)
        label_set_fps: torch.Tensor = torch.cat(fp_tnsrs_from_smi_file)

        # Remove redundancy
        label_set_fps, label_set_smis = remove_redundant_fingerprints(
            label_set_fps, label_set_smis, device
        )

        # Move label_set_fps to cpu # TODO: Why needed?
        label_set_fps = label_set_fps.cpu()

        print(f"Label set size: {len(label_set_fps)}")

        return label_set_fps, label_set_smis

    def _validate_run_inference(
        self: "MoadVoxelModelParent", args: Namespace, ckpt: Optional[str]
    ):
        """Validate the arguments for inference mode.
        
        Args:
            self (MoadVoxelModelParent): This object
            args (Namespace): The user arguments.
            ckpt (Optional[str]): The checkpoint to load.
            
        Raises:
            ValueError: If the arguments are invalid.
        """
        if not ckpt:
            raise ValueError(
                "Must specify a checkpoint (e.g., --load_checkpoint) in inference mode"
            )
        if not args.inference_label_sets:
            raise ValueError(
                "Must specify a label set(s) (--inference_label_sets), which is a comma-separated list of files containing SMILES strings"
            )

        smi_files = [l.strip() for l in args.inference_label_sets.split(",")]
        if (
            "test" in smi_files
            or "train" in smi_files
            or "val" in smi_files
            or "all" in smi_files
        ):
            raise ValueError(
                "Cannot use 'test', 'train', 'val', or 'all' as a label set in inference mode (via --inference_label_sets). Only a comma-separated list of files containing SMILES strings."
            )
        if args.receptor is None:
            raise ValueError("Must specify a receptor (--receptor) in inference mode")
        if args.ligand is None:
            raise ValueError("Must specify a ligand (--ligand) in inference mode")
        if args.branch_atm_loc_xyz is None:
            raise ValueError(
                "Must specify a center (--branch_atm_loc_xyz) in inference mode"
            )

    def run_inference(
        self: "MoadVoxelModelParent",
        args: Namespace,
        ckpt_filename: str,
        save_results_to_disk=True,
    ) -> Union[Dict[str, Any], None]:
        """Run a model on the test and evaluates the output.

        Args:
            self (MoadVoxelModelParent): This object
            args (Namespace): The user arguments.
            ckpt_filename (Optional[str]): The checkpoint to load.
            save_results_to_disk (bool): Whether to save the results to disk.
                If false, the results are returned as a dictionary.

        Returns:
            Union[Dict[str, Any], None]: The results dictionary, or None if
                save_results_to_disk is True.
        """
        self._validate_run_inference(args, ckpt_filename)

        print(
            f"Using the operator {args.aggregation_rotations} to aggregate the inferences."
        )

        pr = cProfile.Profile()
        pr.enable()

        voxel_params = self.init_voxel_params(args)
        device = self.init_device(args)

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

        # Make the voxel
        # cpu = device.type == "cpu"
        cpu = True  # TODO: Required because model.device == "cpu", I think.

        ckpts = [c.strip() for c in ckpt_filename.split(",")]
        fps = []
        for ckpt_filename in ckpts:
            print(f"Using checkpoint {ckpt_filename}")
            model = self.init_model(args, ckpt_filename)

            # TODO: model.device is "cpu". Is that right? Shouldn't it be "cuda"?

            model.eval()

            # You're iterating through multiple checkpoints. This allows output
            # from multiple trained models to be averaged.

            for r in range(args.rotations):
                print(f"    Rotation #{(r + 1)}")

                rot = rand_rot()

                recep_vox = recep.voxelize(
                    voxel_params, cpu=cpu, center=center, rot=rot
                )
                lig_vox = lig.voxelize(voxel_params, cpu=cpu, center=center, rot=rot)

                # Stack the receptor and ligand tensors
                num_features = recep_vox.shape[1] + lig_vox.shape[1]
                dimen1 = lig_vox.shape[2]
                dimen2 = lig_vox.shape[3]
                dimen3 = lig_vox.shape[4]
                vox = torch.cat([recep_vox[0], lig_vox[0]]).reshape(
                    [1, num_features, dimen1, dimen2, dimen3]
                )

                fps.append(model.forward(vox))

        avg_over_ckpts_of_avgs = torch.mean(torch.stack(fps), dim=0)

        # Now get the label sets to use.
        (
            label_set_fingerprints,
            label_set_smis,
        ) = self.create_inference_label_set(
            args, device, [l.strip() for l in args.inference_label_sets.split(",")],
        )

        most_similar = most_similar_matches(
            avg_over_ckpts_of_avgs,
            label_set_fingerprints,
            label_set_smis,
            args.num_inference_predictions,
        )

        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()

        # TODO: Need to understand why this is different, jacob vs. cesar
        output = {
            "most_similar": most_similar[0],
            "fps": {"per_rot": fps, "avg": avg_over_ckpts_of_avgs},
        }

        if not save_results_to_disk:
            # Return the results
            return output

        # If you get here, you are saving the results to disk (default).
        with open(f"{args.default_root_dir}{os.sep}inference_out.smi", "w") as f:
            f.write("SMILES\tScore (Cosine Similarity)\n")
            for smiles, score_cos_similarity in most_similar[0]:
                # [0] because only one prediction

                line = f"{smiles}\t{score_cos_similarity:.3f}"
                print(line)
                f.write(line + "\n")

        # TODO: All this added in jacob branch. Need to remember why.
        output["fps"]["per_rot"] = [v.detach().numpy().tolist() for v in output["fps"]["per_rot"]]
        output["fps"]["avg"] = output["fps"]["avg"].detach().numpy().tolist()
        
        with open(f"{args.default_root_dir}{os.sep}inference_out.tsv", "w") as f:
            f.write("most_similar\t" + json.dumps(output["most_similar"]) + "\n")
            f.write(f"fps_avg\t" + json.dumps(output["fps"]["avg"][0]) + "\n")
            for i, per_rot in enumerate(output["fps"]["per_rot"]):
                f.write(f"fps_rot_{i + 1}\t" + json.dumps(per_rot[0]) + "\n")
            
            # json.dump(output, f)

        

        # TODO: Need to check on some known answers as a "sanity check".

        # TODO: DISCUSS WITH CESAR. Can we add the fragments in most_similar[0] to the parent
        # molecule, to make a composite ligand ready for docking?

