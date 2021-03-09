
import numpy as np
from openbabel import pybel
from torch_geometric.data import Data


def _pybel_bond_index(mol: pybel.Molecule) -> np.ndarray:
    bonds = [mol.OBMol.GetBondById(i) for i in range(mol.OBMol.NumBonds())]
    edges = []
    
    for b in bonds:
        # Atom indexes start at 1.
        start = b.GetBeginAtomIdx() - 1
        end = b.GetEndAtomIdx() - 1
        edges.append((start,end))

    return np.array(edges)


class AtomFeaturizer(object):
    @staticmethod
    def featurize(mol: pybel.Molecule) -> np.ndarray:
        raise NotImplementedError()


class BondFeaturizer(object):
    @staticmethod
    def featurize(mol: pybel.Molecule) -> np.ndarray:
        raise NotImplementedError()


class SimpleAtomFeaturizer(AtomFeaturizer):
    @staticmethod
    def featurize(mol: pybel.Molecule) -> np.ndarray:
        return np.array([x.atomicnum for x in mol.atoms]).reshape(-1,1)


class SimpleBondFeaturizer(BondFeaturizer):
    @staticmethod
    def featurize(mol: pybel.Molecule) -> np.ndarray:
        bonds = [mol.OBMol.GetBondById(i) for i in range(mol.OBMol.NumBonds())]
        return np.array([x.GetBondOrder() for x in bonds]).reshape(-1,1)


class MolGraph(Data):
    """Molecule with 3D coordinates.
    
    Attributes:
    (N atoms) (B bonds)
    - atom_coords: Nx3 array of (x,y,z) coordinates.
    - atom_types: Nx? array of atom feature vectors.
    - bond_index: 2x(B*2) edge_index array.
    - bond_types: (B*2)x? array of bond feature vectors.
    """

    def __init__(self, atom_coords: np.ndarray, atom_types: np.ndarray,
        bond_index: np.ndarray, bond_types: np.ndarray):
        
        self.atom_coords = atom_coords
        self.atom_types = atom_types
        self.bond_index = bond_index
        self.bond_types = bond_types

    def __repr__(self):
        return f'MolGraph(atom_coords={self.atom_coords.shape}, '\
               f'atom_types={self.atom_types.shape}, '\
               f'bond_index={self.bond_index.shape}, '\
               f'bond_types={self.bond_types.shape})'\

    @staticmethod
    def from_pybel(
        mol: pybel.Molecule, 
        af: AtomFeaturizer = SimpleAtomFeaturizer, 
        bf: BondFeaturizer = SimpleBondFeaturizer
    ) -> 'MolGraph':
        atom_coords = np.array([x.coords for x in mol.atoms])
        atom_types = af.featurize(mol)
        bond_index = _pybel_bond_index(mol)
        bond_types = bf.featurize(mol)

        return MolGraph(atom_coords, atom_types, bond_index, bond_types)

    @staticmethod
    def from_sdf(sdf_path: str) -> 'MolGraph':
        raise NotImplementedError()

    @staticmethod
    def from_smiles(
        smiles: str,
        af: AtomFeaturizer = SimpleAtomFeaturizer, 
        bf: BondFeaturizer = SimpleBondFeaturizer
    ) -> 'MolGraph':
        """Create a MolGraph from a smiles string. A reasonable conformation
        is automatically inferred."""

        mol = pybel.readstring('smi', smiles)
        mol.make3D()

        return MolGraph.from_pybel(mol, af, bf)
