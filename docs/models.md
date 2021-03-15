
# Models

This page describes how machine learning models are created in Frag Atlas.

## Quick Start

First create a class that subclasses `atlas.models.base_model.BasePytorchModel`:

```py
from atlas.models.base_model import BasePytorchModel

class MyModel(BasePytorchModel):
    @staticmethod
    def get_params() -> dict:
        ...
    
    @staticmethod
    def build(args: dict) -> dict:
        ...
```

Next, we need to specify two interface methods. `get_params` returns a `dict` containing valid parameters and default arguments for those parameters. These parameters are fixed hyperparameters for the lifetime of the model. For example, we might implement:

```py
@staticmethod
def get_params() -> dict:
    return {
        'input_size': 20,
        'hidden_size': 50,
        'output_size': 5
    }
```

Next, we need to specify how to construct our model architecture given these parameters: `build` returns a dict mapping model names (e.g. `encoder`) to a PyTorch `Module`. We also are provided with an `args` input.

```py
@staticmethod
def build(args: dict) -> dict:
    net = torch.nn.Sequential(
        torch.nn.Linear(args['input_size'], args['hidden_size']),
        torch.nn.ReLU(),
        torch.nn.Linear(args['hidden_size'], args['output_size'])
    )

    return {'net': net}
```

Note: we can initialize multiple model architectures here and they will be bundled together on-disk.

To instantiate our model, we call `create` with an args dict. Any arguments that we leave out will use their default values:

```py
model = MyModel.create({
    'input_size': 5
})

# Models are available on the "models" attribute:
net = model.models['net']
pred = net(torch.zeros((20,5)))
```

We can also save and load these models:
```py
model.save('./path/to/folder')

# Load a duplicate model:
model2 = MyModel.load('./path/to/folder')
```

## Goals

There are several goals that motivate the design choices:

### No train/serve skew
Models should be fully specified on-disk. It should be possible to save and re-load a model without accidentally modifying the architecture or parameters.

All models subclass `atlas.models.base_model.BasePytorchModel` which provides a lightweight framework for building parameterized models.

### Detail-agnostic training scripts

Whenever possible, training scripts should reference as little information as possible in a model. For example, the training script for a molecule autoencoder does not depend on the specific architecture but rather the principle that you can `encode` and `decode` molecules.

Therefore, we can specify a model interface that provides `encode` and `decode` methods and then instance the interface in model-specific files. During the research process, this makes it easy to substitue model architectures.


