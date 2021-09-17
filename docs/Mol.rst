The Mol Class
=============

The :class:`~collagen.data.mol.Mol` class acts as a wrapper over a `Chem.rdmol.RDMol` object and provides various data transformation functions.

Constructing a Mol
------------------

You can construct a Mol using the provided ``from_*`` method or using a data loader (see :doc:`LoadingData`).

.. automethod:: collagen.data.mol.Mol.from_smiles
.. automethod:: collagen.data.mol.Mol.from_rdkit
.. automethod:: collagen.data.mol.Mol.from_prody

Attributes
----------

There are several attribute wrappers for quick access to molecular data:

.. autoproperty:: collagen.data.mol.Mol.has_coords
.. autoproperty:: collagen.data.mol.Mol.coords
.. autoproperty:: collagen.data.mol.Mol.center

.. autoproperty:: collagen.data.mol.Mol.atoms
.. autoproperty:: collagen.data.mol.Mol.mass

.. autoproperty:: collagen.data.mol.Mol.num_atoms
.. autoproperty:: collagen.data.mol.Mol.num_heavy_atoms

.. autoproperty:: collagen.data.mol.Mol.smiles
.. autoproperty:: collagen.data.mol.Mol.iso_smiles

Topology
--------

Graph conversion TODO

.. automethod:: collagen.data.mol.Mol.split_bonds

Voxelization
----------

See :doc:`Voxelization`.
