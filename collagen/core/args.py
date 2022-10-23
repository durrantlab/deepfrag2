import argparse
import json

# It would be tedious to get the user args to some places in the code (e.g.,
# MOAD_target). Let's just make some of the variables globally availble here.
# There arguments are broadly applicable, so it makes sense to separate them
# anyway.

verbose = False


def _add_generic_params(
    parent_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Adds parameters (args) that are common to all collagen apps.

    Args:
        parent_parser (argparse.ArgumentParser): The argparser.

    Returns:
        argparse.ArgumentParser: The updated argparser with the generic
        parameters added.
    """

    parser = parent_parser.add_argument_group("Common")
    parser.add_argument(
        "--cpu",
        default=False,
        # action="store_true"
    )
    parser.add_argument(
        "--wandb_project",
        required=False,
        default=None
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["train", "warm_starting", "test", "lr_finder"],
        help="Can be train, warm_starting, test, or lr_finder. " +
             "If train, trains the model. " +
             "If warm_starting, runs an incremental learning on a new dataset. " +
             "If test, runs inference on the test set. " +
             "If lr_finder, suggests the best learning rate to use. Defaults to train.",
        default="train",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="If specified, the model will be loaded from this checkpoint. You can list multiple checkpoints (separated by commas) for testing/inference.",
    )
    parser.add_argument(
        "--load_newest_checkpoint",
        type=bool,
        required=False,
        default=False,
        # action="store_true",
        help="If set, the most recent checkpoint will be loaded.",
    )
    # TODO: JDD: Load from best validation checkpoint.
    parser.add_argument(
        "--verbose",
        type=bool,
        required=False,
        default=False,
        # action="store_true",
        help="If set, will output additional information during the run. Useful for debugging.",
    )
    parser.add_argument(
        "--json_params",
        required=False,
        default=None,
        help="Path to a json file with parameters that override those specified at the command line.",
    )
    parser.add_argument(
        "--save_params",
        required=False,
        default=None,
        help="Path to a json file where all parameters will be saved. Useful for debugging.",
    )
    parser.add_argument(
        "--learning_rate",
        required=False,
        default=1e-3,
        help="The learning rate.",
    )
    parser.add_argument(
        "--model_for_warm_starting",
        type=str,
        required=False,
        default=None,
        help="Path to .pt file where the model to be used for incremental learning is saved"
    )

    return parent_parser


def _get_arg_parser(
    parser_funcs: list, is_pytorch_lightning=False
) -> argparse.ArgumentParser:
    """Constructs an arg parser.

    Args:
        parser_funcs (list): A list of functions that add arguments to a 
            parser. They each accept a parser and return a parser.
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
    parent_parser = _add_generic_params(parent_parser)

    # Add pytorch lighting parameters if appropriate.
    if is_pytorch_lightning:
        import pytorch_lightning as pl

        pl.Trainer.add_argparse_args(parent_parser)

    return parent_parser


def get_args(parser_funcs = None, post_parse_args_funcs = None, is_pytorch_lightning=False) -> argparse.Namespace:
    """The function creates a parser and gets the associated parameters.

    Args:
        parser_funcs (list, optional): A list of functions that add arguments to
            a parser. They each accept a parser and return a parser. Defaults to
            [].
        fix_args_funcs (list, optional): A list of functions that modify the
            parsed args. Each accepts args and returns args. Allows for
            modifying one argument based on the value of another. Defaults to
            [].
        is_pytorch_lightning (bool, optional): [description]. Defaults to False.

    Returns:
        argparse.Namespace: The parsed and updated args.
    """

    if parser_funcs is None:
        parser_funcs = []
    if post_parse_args_funcs is None:
        post_parse_args_funcs = []

    global verbose

    # Get the parser
    parser = _get_arg_parser(parser_funcs, is_pytorch_lightning)

    # Parse the arguments
    args = parser.parse_args()

    # Make a few select arguments globally available.
    verbose = args.verbose

    # Add parameters from JSON file, which override any command-line parameters.
    if args.json_params:
        with open(args.json_params, "rt") as f:
            new_args = json.load(f)
        for k in new_args.keys():
            setattr(args, k, new_args[k])

    # Fix the arguments.
    for func in post_parse_args_funcs:
        args = func(args)

    # Save all arguments to a json file for debugging.
    if args.save_params is not None:
        with open(args.save_params, "w") as f:
            json.dump(vars(args), f, indent=4)

    # Always print out the arguments to the screen.
    print("\nPARAMETERS")
    print("-----------\n")
    for k, v in vars(args).items():
        print(f"{k.rjust(35)} : {str(v)}")
    print("")

    return args
