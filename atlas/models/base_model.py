
import json
import logging
import os
import pathlib

import torch
from torch import nn
import torch.nn.functional as F


class BasePytorchModel(object):
    """Contains utility code for saving, loading and configuring parameterized
    pytorch models.
    
    Subclasses should implement get_params() to define default/valid arguments
    and build() to implement model construction.

    Model instances can be initialized by calling create/load and can be
    serialized to disk by calling save:

    >>> model = MyModel.create({'param_a': 3, 'param_b': 5})
    >>> model.save('/my/cool/model')
    >>> other = MyModel.load('/my/cool/model')
    """
    args: dict
    models: dict

    def __repr__(self):
        return f'{type(self).__name__}({self.args})'

    @classmethod
    def load(cls, path: str, device='cuda') -> 'BasePytorchModel':
        """Load the model from a directory.
        
        Args:
        - path: Path to saved model directory.
        - device: Device to load model parameters onto. ('cuda' or 'cpu')
        """
        p = pathlib.Path(path)

        with open(p / 'args.json', 'r') as f:
            args = json.load(f)

        mod = cls.create(args)

        for name in mod.models:
            mod.models[name].load_state_dict(
                torch.load(str(p / f'{name}.pt'), map_location=torch.device(device))
            )

        return mod

    def save(self, path: str):
        """Save the model to a given directory."""
        p = pathlib.Path(path)

        os.makedirs(str(p), exist_ok=True)

        with open(p / 'args.json', 'w') as f:
            f.write(json.dumps(self.args))

        for name in self.models:
            torch.save(self.models[name].state_dict(), str(p / f'{name}.pt'))

    @classmethod
    def create(cls, args: dict = {}) -> 'BasePytorchModel':
        """
        Create an instance of a BasePytorchModel with provided arguments.
        
        This will throw an exception if any provided argument is not found in
        the default argument list.
        """
        mod = cls.__new__(cls)

        default_args = cls.get_params()
        BasePytorchModel._check_valid_args(default_args, args)
        default_args.update(args)

        mod.args = default_args
        mod.setup(mod.args)
        mod.models = mod.build(mod.args)

        return mod

    @staticmethod
    def _check_valid_args(default: dict, user: dict):
        invalid_args = {}
        for arg in user:
            if not arg in default:
                invalid_args[arg] = user[arg]

        if len(invalid_args) == 0:
            return

        logging.error(f'Invalid arguments found in specification: {invalid_args} '
                      f'not in the provided list of arguments: {[k for k in default]}')

        raise ValueError()

    # --- Subclasses should implement the following: ---

    @staticmethod
    def get_params() -> dict:
        """Returns a dict of default parameters."""
        return {}

    def setup(self, args: dict):
        """Optional overload-able setup function called once during model
        construction."""
        pass

    @staticmethod
    def build(args: dict) -> dict:
        """Subclasses should use this method to construct pytorch models from
        the internal self.args dict of parameters.
        
        Returns a dict of (str -> pytorch model)
        """
        return {}


class TestModel(BasePytorchModel):
    """An example test model."""

    @staticmethod
    def get_params() -> dict:
        return {
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 2
        }
    
    @staticmethod
    def build(args: dict) -> dict:
        model = nn.Sequential(
            nn.Linear(args['input_size'], args['hidden_size']),
            nn.ReLU(),
            nn.Linear(args['hidden_size'], args['output_size'])
        )

        return {'model': model}
