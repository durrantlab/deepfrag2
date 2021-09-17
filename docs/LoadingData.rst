Loading Data
============

Data loading is provided by a series of Datasets that implement :class:`collagen.data.mol.MolDataset` (which is itself a child of :class:`torch.utils.data.Dataset`.

ZINC Database
-------------

Homepage: https://zinc.docking.org/

.. autoclass:: collagen.data.zinc.ZINCDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: collagen.data.zinc.ZINCDatasetH5
    :members:
    :undoc-members:
    :show-inheritance:

MOAD
----

:doc:`external/Moad`

