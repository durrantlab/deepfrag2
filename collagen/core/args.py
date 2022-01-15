import argparse
import json


def _get_arg_parser(parser_funcs: list, is_pytorch_lightning=False):
    """Constructs an arg parser.

    Args:
        parser_funcs (list): A list of functions that add arguments to a 
            parser. They each return a parser.
        is_pytorch_lightning (bool, optional): Whether the app uses pytorch
            lightning. Defaults to False.

    Returns:
        argparse.ArgumentParser: A parser with all the arguments added.
    """
    
    # Create the parser
    parent_parser = argparse.ArgumentParser()

    # Add arguments to it per each input function
    for func in parser_funcs:
        parent_parser = func(parent_parser)

    # Add arguments that are common to all apps.
    parser = parent_parser.add_argument_group("Common")
    parser.add_argument(
        "--json_params",
        required=False,
        default=None,
        help="Path to a json file with parameters that override those specified at the command line.",
    )

    # Add pytorch lighting parameters if appropriate.
    if is_pytorch_lightning:
        import pytorch_lightning as pl

        parser = pl.Trainer.add_argparse_args(parent_parser)

    return parent_parser


def get_args(parser_funcs: list, is_pytorch_lightning=False) -> argparse.Namespace:
    # Get the parser
    parser = _get_arg_parser(parser_funcs, is_pytorch_lightning)

    # Parser the arguments
    args = parser.parse_args()

    # Add parameters from JSON file.
    if args.json_params:
        with open(args.json_params, "rt") as f:
            new_args = json.load(f)
            for k in new_args.keys():
                setattr(args, k, new_args[k])

    # Always print out the arguments to the screen.
    print("\nPARAMETERS")
    print("-----------\n")
    for k, v in vars(args).items():
        print(k.rjust(35) + " : " + str(v))
    print("")

    return args
