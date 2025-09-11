"""Runs DeepFrag2 in a simplified inference-only mode."""
import sys
import os
import shutil
import glob
from collagen.main import main as run_main


def main():
    """Entry point for the simplified deepfrag2_inference script."""
    usage_example = """
====================================================================================
 deepfrag2: Simple command-line tool for fragment generation (deepfrag2 inference).
====================================================================================

Example usage:

deepfrag2 \\
    --receptor path/to/receptor.pdb \\
    --ligand path/to/ligand.sdf \\
    --branch_atm_loc_xyz "x,y,z" \\
    --out path/to/results.tsv

By default, this script uses the 'gte_4_best' model and the 'gte_4_all' fragment
set. You can specify others with --load_checkpoint and --inference_label_sets.
See README.md for details.

For advanced usage, please use the deepfrag2full script.

--------------------------------------------------------------------------------
"""
    print(usage_example)

    hardcoded_args_map = {
        '--mode': 'inference_single_complex',
        '--default_root_dir': './',
        '--cache': 'None',
        "--cpu": True,
    }

    # Manually parse for '--out' and filter it out
    original_user_args = sys.argv[1:]
    out_path = None
    args_for_main: list[str] = []
    i = 0
    while i < len(original_user_args):
        arg = original_user_args[i]
        if arg == '--out':
            if i + 1 < len(original_user_args):
                out_path = original_user_args[i + 1]
                i += 1  # Skip the value
        elif arg.startswith('--out='):
            out_path = arg.split('=', 1)[1]
        else:
            args_for_main.append(arg)
        i += 1

    # Check if user provided arguments that we need to default
    load_checkpoint_provided = any(arg.startswith('--load_checkpoint') for arg in args_for_main)
    inference_label_sets_provided = any(arg.startswith('--inference_label_sets') for arg in args_for_main)

    # Construct the new sys.argv
    final_argv = [sys.argv[0]]

    # Add hardcoded arguments
    for key, value in hardcoded_args_map.items():
        if isinstance(value, bool):
            if value:
                final_argv.append(key)
        else:
            final_argv.extend([key, str(value)])
    # Add default for --load_checkpoint if not provided by user
    if not load_checkpoint_provided:
        final_argv.extend(['--load_checkpoint', 'gte_4_best'])

    # Add default for --inference_label_sets if not provided by user
    if not inference_label_sets_provided:
        final_argv.extend(['--inference_label_sets', 'gte_4_all'])

    # Add the rest of the user's arguments (without --out)
    final_argv.extend(args_for_main)

    sys.argv = final_argv

    try:
        run_main()
    finally:
        output_dir_base = "predictions_Single_Complex"
        if os.path.isdir(output_dir_base):
            if out_path:
                try:
                    # Use glob to find the single .tsv file
                    search_pattern = os.path.join(output_dir_base, '**', '*.tsv')
                    found_files = glob.glob(search_pattern, recursive=True)

                    if len(found_files) == 1:
                        source_file_path = found_files[0]
                        dest_dir = os.path.dirname(out_path)
                        if dest_dir:
                            os.makedirs(dest_dir, exist_ok=True)
                        shutil.move(source_file_path, out_path)
                        print(f"\nOutput saved to {out_path}")
                    elif len(found_files) == 0:
                        print(f"\nWarning: No output .tsv file was found in '{output_dir_base}' to move.", file=sys.stderr)
                    else:
                        print(f"\nWarning: Multiple .tsv files found in '{output_dir_base}'. Could not determine which one to move.", file=sys.stderr)

                except Exception as e:
                    print(f"\nError moving output file: {e}", file=sys.stderr)

            # Clean up the directory regardless
            shutil.rmtree(output_dir_base)

if __name__ == "__main__":
    main()