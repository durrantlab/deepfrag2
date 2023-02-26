# In order to enable both command-line and API use, we need to define default
# values for the arguments independently of the argument parser. This is done by
# defining a function that returns a dictionary of default values. The function
# is called by the argument parser to set the default values and by the API to
# get the default values.

# I'm only going to define defaults here for select arguments. The rest will be
# defined by the argparser or must be explicitly defined via the API.

from typing import Any, Dict


def get_default_args() -> Dict[str, Any]:
    return {
        "fragment_representation": "rdk10iiii",
    }
