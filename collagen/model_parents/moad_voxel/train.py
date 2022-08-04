from argparse import Namespace
from typing import Any, Type, TypeVar, List, Optional, Tuple, Dict
from torchinfo import summary
import json
from tqdm.std import tqdm
from collagen.external import MOADInterface
from collagen.external.moad.split import compute_moad_split

class MoadVoxelModelTrain(object):
    def _get_frag_counts(self, args: Namespace):
        # Load cache json.
        with open(args.cache, "r") as f:
            cache = json.load(f)

        frag_counts = {}
        for recep_name in tqdm(cache, desc="Counting fragment SMILES..."):
            for lig_id in cache[recep_name]:
                for frag_smile in cache[recep_name][lig_id]["frag_smiles"]:
                    if frag_smile not in frag_counts:
                        frag_counts[frag_smile] = 0
                    frag_counts[frag_smile] += 1
        return frag_counts


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

        # Get just the fragment counts (for weighted averaging). Note that I never
        # got this to work, but leaving it here in case you return to this in the
        # future... TODO: Should be a flag?
        frag_counts = self._get_frag_counts(args)

        # pr.disable()
        # s = StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        # ps.print_stats()
        # open('cProfilez.txt', 'w+').write(s.getvalue())

        train_data = self.get_data_from_split(args, moad, train, voxel_params, device)
        val_data = self.get_data_from_split(args, moad, val, voxel_params, device)

        model = self.init_model(args, ckpt, frag_counts)

        model_stats = summary(model, (16, 10, 24, 24, 24), verbose=0)
        summary_str = str(model_stats)
        print(summary_str)

        trainer.fit(model, train_data, val_data, ckpt_path=ckpt)

        self._save_used(model, args)
