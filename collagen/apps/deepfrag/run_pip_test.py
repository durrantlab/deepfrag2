"""Runs a test case for DeepFrag2."""
import sys
import os
from InferenceDF2 import main as deepfrag2_main


def main():
    """Entry point for the deepfrag2_test script."""
    # Use a fixed output file name as in the user's example command.
    out_path = "tmp.tsv"

    # Get the directory where this script is located. This is a robust way to
    # find package data files relative to the script's location.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths to the example data files.
    receptor_path = os.path.join(script_dir, 'example_data', '5VUP_prot_955.pdb')
    ligand_path = os.path.join(script_dir, 'example_data', '5VUP_lig_955.sdf')


    print("Running DeepFrag2 test case...")
    print(f"Receptor: {receptor_path}")
    print(f"Ligand: {ligand_path}")
    print(f"Output will be saved to: {os.path.abspath(out_path)}")

    # Temporarily replace sys.argv
    original_argv = sys.argv
    sys.argv = [
        'deepfrag2',
        '--receptor', receptor_path,
        '--ligand', ligand_path,
        '--branch_atm_loc_xyz', '12.413000,3.755000,59.021999',
        '--out', out_path
    ]

    try:
        deepfrag2_main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

    with open(os.path.abspath(out_path), 'r') as f:
        lines = f.readlines()
        if not "SMILES" in lines[0]:
            print("Test failed: Output file does not contain expected header.")
            # Remove output file before exiting
            if os.path.exists(out_path):
                os.remove(out_path)
            sys.exit(1)
        if len(lines) < 2:
            print("Test failed: Output file does not contain expected data.")
            # Remove output file before exiting
            if os.path.exists(out_path):
                os.remove(out_path)
            sys.exit(1)

    print(f"\nTest finished successfully.")
    if os.path.exists(out_path):
        os.remove(out_path)
    print("Temporary output file removed.")

if __name__ == "__main__":
    main()