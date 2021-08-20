The Mol Class
=============

The :class:`~atlas.data.mol.Mol` class acts as a wrapper over a `Chem.rdmol.RDMol` object and provides various data transformation functions.

Constructing a Mol
------------------

You can construct a Mol using the provided ``from_*`` method or using a data loader (see :doc:`LoadingData`).

.. automethod:: atlas.data.mol.Mol.from_smiles
.. automethod:: atlas.data.mol.Mol.from_rdkit

Attributes
----------

There are several attribute wrappers for quick access to molecular data:


.. autoproperty:: atlas.data.mol.Mol.has_coords
.. autoproperty:: atlas.data.mol.Mol.coords
.. autoproperty:: atlas.data.mol.Mol.center

.. autoproperty:: atlas.data.mol.Mol.atoms
.. autoproperty:: atlas.data.mol.Mol.mass

.. autoproperty:: atlas.data.mol.Mol.num_atoms
.. autoproperty:: atlas.data.mol.Mol.num_heavy_atoms

.. autoproperty:: atlas.data.mol.Mol.smiles
.. autoproperty:: atlas.data.mol.Mol.iso_smiles

Topology
--------

Graph conversion TODO

.. automethod:: atlas.data.mol.Mol.split_bonds

Voxelization
----------

See :doc:`Voxelization`.
