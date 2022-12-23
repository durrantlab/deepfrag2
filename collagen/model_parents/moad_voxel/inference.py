from argparse import ArgumentParser, Namespace
import cProfile
from io import StringIO
import pstats
from collagen.model_parents.moad_voxel.test_inference_utils import (
    remove_redundant_fingerprints,
)
import torch
from typing import Any, List, Optional, Tuple
from collagen.core.molecules.mol import Mol, mols_from_smi_file
from collagen.external.moad.types import Entry_info
from collagen.metrics.metrics import most_similar_matches
import prody
import numpy as np
from collagen.util import rand_rot
import rdkit
from rdkit import Chem
import pytorch_lightning as pl


class MoadVoxelModelInference(object):
    def _create_inference_label_set(
        self: "MoadVoxelModelParent",
        args: ArgumentParser,
        device: Any,
        smi_files: List[str],
    ) -> Tuple[torch.Tensor, List[Entry_info]]:
        """Creates a label set (look-up) tensor and smiles list for testing.
        Can be comprised of the fingerprints in the train and/or test and/or
        val sets, as well as SMILES strings from a file.

        Args:
            self (MoadVoxelModelParent): This object
            args (Namespace): The user arguments.
            device (Any): The device to use.
            smi_files (List[str]): The file(s) containing SMILES strings.

        Returns:
            Tuple[torch.Tensor, List[Entry_info]]: The updated fingerprint
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
                        mol.fingerprint("rdk10", args.fp_size),
                        dtype=torch.float32,
                        device=device,
                        requires_grad=False,
                    ).reshape((1, args.fp_size))
                )
                label_set_smis.append(smi)
        label_set_fps = torch.cat(fp_tnsrs_from_smi_file)

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
        if args.xyz is None:
            raise ValueError("Must specify a center (--xyz) in inference mode")

    def run_inference(
        self: "MoadVoxelModelParent", args: Namespace, ckpt: Optional[str]
    ):
        # Runs a model on the test and evaluates the output.

        self._validate_run_inference(args, ckpt)

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
        center = np.array([float(v.strip()) for v in args.xyz.split(",")])

        # Load the ligand
        suppl = Chem.SDMolSupplier(str(args.ligand))
        rdmol = [x for x in suppl if x is not None][0]
        lig = Mol.from_rdkit(rdmol, strict=False)

        # Make the voxel
        # cpu = device.type == "cpu"
        cpu = True  # TODO: Required because model.device == "cpu", I think.

        ckpts = [c.strip() for c in ckpt.split(",")]
        fps = []
        for ckpt in ckpts:
            print(f"Using checkpoint {ckpt}")
            model = self.init_model(args, ckpt)

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
            label_set_entry_infos,
        ) = self._create_inference_label_set(
            args,
            device,
            [l.strip() for l in args.inference_label_sets.split(",")],
        )

        most_similar = most_similar_matches(
            avg_over_ckpts_of_avgs,
            label_set_fingerprints,
            label_set_entry_infos,
            25,  # self.NUM_MOST_SIMILAR_PER_ENTRY,
        )

        for smiles, score in most_similar[0]:
            # [0] because only one prediction

            score = 1.0 - score  # Bigger score better
            print(f"{score:.3f} {smiles}")

        # TODO: Need to check on some known answers as a "sanity check".
        
        # TODO: Can we add the fragments in most_similar[0] to the parent
        # molecule, to make a composite ligand ready for docking?

        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()
