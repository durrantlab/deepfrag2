Binding MOAD
============

Overview
--------

The Binding MOAD download provides a set of PDB structures for given targets (ABCD.bio1, ABCD.bio2, ..., ABCD.bioN) and a metadata file (every.csv) that contains information on protein families, ligands, and binding affinities.

The core interface for this data is :class:`collagen.data.moad.MOADBase`.

.. code-block:: python

    from collagen.data.moad import MOADBase

    moad = MOADBase(
        metadata='/path/to/every.csv',
        structures='/path/to/BindingMoad2019'
    )

View targets:

.. code-block:: python

    >>> print(moad.targets[:10])
    ['1fwe', '6h8j', '5ol4', '4ubp', '4ac7', '4cex', '3ubp', '4s3f', '4g9p', '4s3c']


Fetch a specific target:

.. code-block:: python

    >>> moad['11gs']
    MOAD_target(pdb_id='11GS', ligands=[MOAD_ligand(name='GSH EAA:A:210', validity='valid', affinity_measure='Ki', affinity_value='1.5', affinity_unit='uM', smiles='Clc1c(Cl)c(C(=O)C(CSCC(NC(=O)CCC([N+H3])C(=O)[O-])C(=O)NCC(=O)[O-])CC)ccc1OCC(=O)[O-]'), MOAD_ligand(name='GSH EAA:B:210', validity='valid', affinity_measure='Ki', affinity_value='1.5', affinity_unit='uM', smiles='Clc1c(Cl)c(C(=O)C(CSCC(NC(=O)CCC([N+H3])C(=O)[O-])C(=O)NCC(=O)[O-])CC)ccc1OCC(=O)[O-]'), MOAD_ligand(name='MES:A:212', validity='invalid', affinity_measure='', affinity_value='', affinity_unit='', smiles='C1COCC[NH+]1CCS(=O)(=O)[O-]'), MOAD_ligand(name='MES:B:212', validity='invalid', affinity_measure='', affinity_value='', affinity_unit='', smiles='C1COCC[NH+]1CCS(=O)(=O)[O-]')], files=[PosixPath('/Users/Harrison/cbio/data/moad/full/BindingMOAD_2020_b/11gs.bio1')])

Core Documentation
------------------

.. autoclass:: collagen.data.moad.MOADBase
    :members:
    :undoc-members:
    :special-members: __getitem__

.. autoclass:: collagen.data.moad.MOAD_class
    :members:
    :undoc-members:

.. autoclass:: collagen.data.moad.MOAD_family
    :members:
    :undoc-members:

.. autoclass:: collagen.data.moad.MOAD_target
    :members:
    :undoc-members:
    :special-members: __len__, __getitem__

.. autoclass:: collagen.data.moad.MOAD_ligand
    :members:
    :undoc-members:

.. autoclass:: collagen.data.moad.MOAD_split
    :members:
    :undoc-members:

Datasets
--------

Terminal Fragment Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`collagen.data.moad.MOADFragmentDataset` class provides a 
