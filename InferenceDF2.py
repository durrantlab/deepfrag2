"""Runs DeepFrag2 in a simplified inference-only mode."""
import sys
from collagen.main import main as run_main


def main():
    """Entry point for the simplified deepfrag2_inference script."""
    usage_example = """
================================================================================
 deepfrag2_inference: A simplified command-line tool for fragment generation.
================================================================================

Example usage:

deepfrag2_inference \\
    --receptor path/to/receptor.pdb \\
    --ligand path/to/ligand.sdf \\
    --branch_atm_loc_xyz "x,y,z" \\
    --load_checkpoint gte_4_best \\
    --inference_label_sets path/to/fragments.smi

For more advanced options, please use the `deepfrag2cpu` command.
--------------------------------------------------------------------------------
"""
    print(usage_example)

    hardcoded_args_map = {
        '--mode': 'inference_single_complex',
        '--default_root_dir': './',
        '--cache': None,
    }

    # Check if user provided --load_checkpoint
    load_checkpoint_provided = any(arg.startswith('--load_checkpoint') for arg in sys.argv[1:])

    # Filter existing sys.argv to remove hardcoded args
    filtered_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        arg_key = arg.split('=', 1)[0]

        if arg_key in hardcoded_args_map:
            # If it's a key-value pair like --key value, skip next element as well
            if '=' not in arg and i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                i += 1
        else:
            filtered_argv.append(arg)
        i += 1
    
    # Construct the new sys.argv by prepending hardcoded args
    final_argv = [filtered_argv[0]]
    for key, value in hardcoded_args_map.items():
        final_argv.append(key)
        final_argv.append(value)
    
    # Add default for --load_checkpoint if not provided by user
    if not load_checkpoint_provided:
        final_argv.extend(['--load_checkpoint', 'gte_4_best'])

    final_argv.extend(filtered_argv[1:])
    
    sys.argv = final_argv
    run_main()


if __name__ == "__main__":
    main()