"""__init__.py"""

__all__ = [
    "Mol",
    "BackedMol",
    "DelayedMolVoxel",
    "AbstractMol",
    "AbstractAtom",
    "AbstractBond",
    "VoxelParams",
    "VoxelParamsDefault",
    "AtomFeaturizer",
    "AtomicNumFeaturizer",
    "AnyAtom",
    "MultiLoader",
    "GraphMol",
]

from .molecules.mol import Mol, BackedMol, DelayedMolVoxel
from .molecules.abstract_mol import AbstractMol, AbstractAtom, AbstractBond
from .voxelization.voxelizer import VoxelParams, VoxelParamsDefault
from .molecules.atom_featurizer import AtomFeaturizer, AtomicNumFeaturizer
from .types import AnyAtom
from .loader import MultiLoader

try:
    from .molecules.graph_mol import GraphMol
except Exception:
    GraphMol = None
    print("collagen.GraphMol requires torch_geometric!")
