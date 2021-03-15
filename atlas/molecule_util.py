
import numpy as np
from openbabel import pybel
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def _pybel_bond_index(mol: pybel.Molecule) -> np.ndarray:
    bonds = [mol.OBMol.GetBondById(i) for i in range(mol.OBMol.NumBonds())]
    edges = []
    
    for b in bonds:
        # Atom indexes start at 1.
        start = b.GetBeginAtomIdx() - 1
        end = b.GetEndAtomIdx() - 1
        edges.append((start,end))

    return np.array(edges).transpose()


class AtomFeaturizer(object):
    @staticmethod
    def featurize(mol: pybel.Molecule) -> np.ndarray:
        raise NotImplementedError()


class BondFeaturizer(object):
    @staticmethod
    def featurize(mol: pybel.Molecule) -> np.ndarray:
        raise NotImplementedError()


class SimpleAtomFeaturizer(AtomFeaturizer):
    # Most common ligand atoms.
    ATOMS = [6,8,7,16,9,15,17,35,5,53,34,26,27,44,14,29,33,45,4,30,77,12,75,23,51,76,80]

    @staticmethod
    def featurize(mol: pybel.Molecule) -> np.ndarray:
        atomnum = [x.atomicnum for x in mol.atoms]

        # idx 0 represents "other"
        atom_types = np.zeros((len(atomnum), len(SimpleAtomFeaturizer.ATOMS)+1))
        for i in range(len(atomnum)):
            if atomnum[i] in SimpleAtomFeaturizer.ATOMS:
                atom_types[i][SimpleAtomFeaturizer.ATOMS.index(atomnum[i])+1] = 1
            else:
                atom_types[i][0] = 1

        return atom_types


class SimpleBondFeaturizer(BondFeaturizer):
    @staticmethod
    def featurize(mol: pybel.Molecule) -> np.ndarray:
        bonds = [mol.OBMol.GetBondById(i) for i in range(mol.OBMol.NumBonds())]

        # inside () is mutually exclusive classes
        # [(aro, single, double, triple), in_ring]
        bond_types = np.zeros((len(bonds), 5))
        for i in range(len(bonds)):
            bond = bonds[i]

            if bond.IsAromatic():
                bond_types[i][0] = 1
            else:
                order = bond.GetBondOrder()

                if order == 1: bond_types[i][1] = 1
                elif order == 2: bond_types[i][2] = 1
                elif order == 3: bond_types[i][3] = 1
            
            if bond.IsInRing():
                bond_types[i][4] = 1

        return bond_types


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
        
        self.atom_coords = torch.tensor(atom_coords, dtype=torch.float)
        self.atom_types = torch.tensor(atom_types, dtype=torch.float)
        self.bond_index = torch.tensor(bond_index, dtype=torch.long)
        self.bond_types = torch.tensor(bond_types, dtype=torch.float)

        # MolGraph objects may contain a smiles string if initialized with
        # MolGraph.from_smiles().
        self.smiles: str = None

        # An optional dict with metadata information about the origin of the
        # MolGraph object. E.g. may contain a "ZINC" indentifier.
        self.meta: dict = {}

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
        bf: BondFeaturizer = SimpleBondFeaturizer,
        make_3D: bool = True
    ) -> 'MolGraph':
        """Create a MolGraph from a smiles string. A reasonable conformation
        is automatically inferred."""

        mol = pybel.readstring('smi', smiles)
        if make_3D:
            mol.make3D()
        mol.removeh()

        m = MolGraph.from_pybel(mol, af, bf)
        m.smiles = smiles
        m._make_undirected()
        return m

    def _make_undirected(self):
        """Duplicate bond_index and bond_types so that each edge is listed as
        (a,b) and (b,a). This format is useful for message passing convolutions."""
        self.bond_index = torch.cat(
            (self.bond_index, torch.flip(self.bond_index, (0,))), 1)

        self.bond_types = torch.cat(
            (self.bond_types, self.bond_types), 0)

    def to_sdf(self) -> str:
        """Convert the molecule to sdf using atom_coords.
        
        This method requires the molecule to have a smiles string set.
        """
        assert self.smiles is not None

        mol = pybel.readstring('smi', self.smiles)
        for i in range(len(mol.atoms)):
            mol.atoms[i].OBAtom.SetVector(
                *[float(x) for x in self.atom_coords[i]])
        return mol.write('sdf')


class MolGraphProvider(Dataset):
    """Abstract interface for a MolGraphProvider object.
    
    Subclasses should implement __len__ and __getitem__ to support enumeration
    over the data.
    """

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx) -> MolGraph:
        raise NotImplementedError()
