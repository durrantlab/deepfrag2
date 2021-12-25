Loading Data
============

Collagen provides a series of external dataset adapters. There are two types of adapters:

* **Interfaces** provide scriptable access to the contents of an on-disk dataset, pulling out metadata and transforming structures into :class:`~collagen.core.mol.Mol` objects.
* **Datasets** subclass :class:`torch.utils.data.Dataset` and implement post-processing (such as voxelization), generating train/val/test splits and iterating over examples for a given interface.

Supported Datasets
------------------

The following classes are importable under the ``collagen.external`` namespace:

ZINC
^^^^

Homepage: https://zinc.docking.org/

...

Binding MOAD
^^^^^^^^^^^^

Homepage: http://www.bindingmoad.org/

Interfaces:

.. autosummary::
    ~collagen.external.moad.MOADInterface

Datasets:

.. autosummary::
    ~collagen.external.moad.fragment.MOADFragmentDataset
    ~collagen.external.moad.MOADPocketDataset
