The Mol Class
=============

Molecular structures in Collagen are stored in the :class:`~collagen.Mol` class. There are several variants depending on the type and source of molecular data:

* :class:`~collagen.BackedMol`: A Mol backed by an RDKit :class:`rdkit.Chem.rdchem.Mol`. Used for representing *real* molecular data (i.e. loaded from a dataset or constructed from SMILES strings).
* :class:`~collagen.AbstractMol`: A customizeable Mol that does not care about chemical feasability constraints. Used for representing graph-like molecular information in the same coordinate space as a :class:`~collagen.BackedMol`.

Constructing a Mol
------------------

You can construct a Mol using the provided ``from_*`` method or using an external data loader (see :doc:`LoadingData`).

.. currentmodule:: collagen
.. autosummary::
    Mol.from_smiles
    Mol.from_rdkit
    Mol.from_prody

The :class:`~collagen.AbstractMol` is intended to be created programatically using the following two methods:

.. currentmodule:: collagen
.. autosummary::
    AbstractMol.add_atom
    AbstractMol.add_bond

For example:

.. code-block:: python

    mol = AbstractMol()
    a = mol.add_atom(AbstractAtom(coord=[1,2,3]))
    b = mol.add_atom(AbstractAtom(coord=[5,6,7]))
    mol.add_bond(AbstractBond(edge=(a,b)))

Attributes
----------

There are several attribute wrappers for quick access to molecular data:

.. currentmodule:: collagen
.. autosummary::
    Mol.coords
    Mol.center
    Mol.atoms
    Mol.mass
    Mol.num_atoms
    Mol.num_heavy_atoms
    Mol.smiles

Topology
--------

Graph conversion TODO

.. automethod:: collagen.Mol.split_bonds

Voxelization
------------

See :doc:`Voxelization`.
