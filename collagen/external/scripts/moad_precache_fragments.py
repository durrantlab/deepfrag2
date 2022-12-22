import argparse

from collagen.external.moad.fragment import MOADFragmentDataset
from collagen.external.moad.moad_interface import MOADInterface


# Disable warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import prody

prody.confProDy(verbosity="none")


def run(args):
    # Not sure this used. But if it is, moad = MOADInterface(args.every_csv,
    # args.data_dir, args.cache_pdbs_to_disk) missing params
    print("HEREHEREHERE???")  
    import pdb; pdb.set_trace()
    moad = MOADInterface(args.every_csv, args.data_dir, args.cache_pdbs_to_disk)
    dat = MOADFragmentDataset(moad, cache_file=args.out, cache_cores=args.cores)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--every_csv", required=True, help="Path to MOAD every.csv")
    parser.add_argument(
        "--data_dir", required=True, help="Path to MOAD root structure folder"
    )
    parser.add_argument("--out", required=True, help="Path to output cache.json file")
    parser.add_argument("--cores", type=int, default=1, help="Number of cores to use")
    args = parser.parse_args()
    run(args)
