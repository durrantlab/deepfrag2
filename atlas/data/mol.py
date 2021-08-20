from dataclasses import dataclass
from typing import List, Tuple, Iterator

import numpy as np
import rdkit.Chem.AllChem as Chem
from rdkit.Chem.Descriptors import ExactMolWt
import torch
from torch.utils.data import Dataset

from .fingerprints import fingerprint_for
from .voxelizer import numba_ptr, mol_gridify


class Mol(object):
    """
    A Mol is a thin wrapper over rdkit.Chem.rdchem.Mol that provides data
    transformation utilities.
    """

    def __init__(self, rdmol: "rdkit.Chem.rdchem.Mol" = None):
        """Initialize a new Mol, optionally backed by an RDKit mol."""
        self.rdmol = rdmol

        # An optional dict with metadata information about the origin of the
        # Mol object. E.g. may contain a "ZINC" indentifier.
        self.meta: dict = {}

    def __repr__(self):
        if "zinc_id" in self.meta:
            return f'Mol({self.meta["zinc_id"]})'
        else:
            return f'Mol(smiles="{self.smiles}")'

    @staticmethod
    def from_smiles(
        smiles: str, sanitize: bool = False, make_3D: bool = False
    ) -> "Mol":
        """Construct a Mol from a SMILES string.

        Args:
            smiles (str): A SMILES string.
            sanitize (bool, optional): If True, attempt to sanitize the internal RDKit molecule.
            make_3D (bool, optional): If True, generate 3D coordinates.

        Returns:
            atlas.data.mol.Mol: A new Mol object.

        Examples:
            Load aspirin from a SMILES string:

            >>> Mol.from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')
            Mol(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        """
        rdmol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        rdmol.UpdatePropertyCache()
        if make_3D:
            Chem.EmbedMolecule(rdmol)

        return Mol(rdmol)

    @staticmethod
    def from_rdkit(rdmol: "rdkit.Chem.rdchem.Mol") -> "Mol":
        """Construct a Mol from an RDKit Mol.

        Args:
            rdmol (rdkit.Chem.rdchem.Mol): An existing RDKit Mol.

        Returns:
            atlas.data.mol.Mol: A new Mol object.
        """
        rdmol.UpdatePropertyCache()
        return Mol(rdmol)

    @property
    def smiles(self) -> str:
        """Convert the internal rdmol to a SMILES string.

        Note:
            This version returns a non-isomeric SMILES.
        """
        return Chem.MolToSmiles(self.rdmol, isomericSmiles=False)

    @property
    def iso_smiles(self) -> str:
        """Convert the internal rdmol to a SMILES string.

        Note:
            This version returns isomeric SMILES.
        """
        return Chem.MolToSmiles(self.rdmol, isomericSmiles=True)

    @property
    def has_coords(self) -> bool:
        return len(self.rdmol.GetConformers()) > 0

    @property
    def coords(self) -> "np.array":
        """Atomic coordinates as a NumPy array."""
        return self.rdmol.GetConformer().GetPositions()

    @property
    def center(self) -> "np.array":
        return np.mean(self.coords, axis=0)

    @property
    def atoms(self) -> List["rdkit.Chem.rdchem.Atom"]:
        return list(self.rdmol.GetAtoms())

    @property
    def mass(self) -> float:
        return ExactMolWt(self.rdmol)

    @property
    def num_atoms(self) -> int:
        return self.rdmol.GetNumAtoms()

    @property
    def num_heavy_atoms(self) -> int:
        return self.rdmol.GetNumHeavyAtoms()

    def split_bonds(
        self, only_single_bonds: bool = True
    ) -> Iterator[Tuple["Mol", "Mol"]]:
        """
        Iterate over all bonds in the Mol and try to split into two fragments, returning tuples of produced fragments.

        Args:
            only_single_bonds (bool): If True (default) only cut on single bonds.

        Examples:
            >>> mol = Mol.from_smiles('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O')
            >>> for a,b in mol.split_bonds():
            ...   print(a)
            ...   print(b)
            Mol(smiles="*C")
            Mol(smiles="*C(C)CC1=CC=C(C(C)C(=O)O)C=C1")
            Mol(smiles="*C(C)CC1=CC=C(C(C)C(=O)O)C=C1")
            Mol(smiles="*C")
            Mol(smiles="*C(C)C")
            Mol(smiles="*CC1=CC=C(C(C)C(=O)O)C=C1")
            Mol(smiles="*CC(C)C")
            Mol(smiles="*C1=CC=C(C(C)C(=O)O)C=C1")
            Mol(smiles="*C1=CC=C(CC(C)C)C=C1")
            Mol(smiles="*C(C)C(=O)O")
            Mol(smiles="*C(C(=O)O)C1=CC=C(CC(C)C)C=C1")
            Mol(smiles="*C")
            Mol(smiles="*C(C)C1=CC=C(CC(C)C)C=C1")
            Mol(smiles="*C(=O)O")
            Mol(smiles="*C(=O)C(C)C1=CC=C(CC(C)C)C=C1")
            Mol(smiles="*O")
        """
        num_mols = len(Chem.GetMolFrags(self.rdmol, asMols=True, sanitizeFrags=False))
        assert (
            num_mols == 1
        ), f"Error, calling split_bonds() on a Mol with {num_mols} parts."

        for i in range(self.rdmol.GetNumBonds()):
            # Filter single bonds.
            if (
                only_single_bonds
                and self.rdmol.GetBondWithIdx(i).GetBondType()
                != Chem.rdchem.BondType.SINGLE
            ):
                continue

            split_mol = Chem.rdmolops.FragmentOnBonds(self.rdmol, [i])
            fragments = Chem.GetMolFrags(split_mol, asMols=True, sanitizeFrags=False)

            # Skip if this did not break the molecule into two pieces.
            if len(fragments) != num_mols + 1:
                continue

            # Ensure the fragment has at least one heavy atom.
            if fragments[1].GetNumHeavyAtoms() == 0:
                continue

            yield (Mol.from_rdkit(fragments[0]), Mol.from_rdkit(fragments[1]))

    def graph(self):
        """TODO"""
        raise NotImplementedError()

    def voxelize(
        self,
        params: "VoxelParams",
        cpu: bool = False,
        center: "np.ndarray" = None,
        rot: "np.ndarray" = np.array([0, 0, 0, 1]),
    ) -> "torch.Tensor":
        """
        Convert a Mol to a voxelized tensor.

        Example:
            >>> m = Mol.from_smiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', make_3D=True)
            >>> vp = VoxelParams(
            ...     resolution=0.75,
            ...     width=24,
            ...     atom_featurizer=AtomicNumFeaturizer([1,6,7,8,16])
            ... )
            >>> tensor = m.voxelize(vp, cpu=True)
            >>> print(tensor.shape)
            torch.Size([1, 5, 24, 24, 24])

        Args:
            params (VoxelParams): Voxelation parameter container.
            cpu (bool): If True, run on the CPU, otherwise use CUDA.
            center: (np.ndarray): Optional, if set, center the grid on this 3D coordinate.
            rot: (np.ndarray): A size 4 array describing a quaternion rotation for the grid.
        """
        params.validate()

        tensor = torch.zeros(size=params.tensor_size())
        if not cpu:
            tensor = tensor.cuda()

        self.voxelize_into(
            tensor, batch_idx=0, center=center, params=params, cpu=cpu, rot=rot
        )

        return tensor

    def voxelize_into(
        self,
        tensor: "torch.Tensor",
        batch_idx: int,
        params: "VoxelParams",
        cpu: bool = False,
        layer_offset: int = 0,
        center: "np.ndarray" = None,
        rot: "np.ndarray" = np.array([0, 0, 0, 1]),
    ):
        """
        Voxelize a Mol into an existing 5-D tensor.

        Example:
            >>> smi = [
            ...     'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            ...     'CC(=O)OC1=CC=CC=C1C(=O)O',
            ...     'CCCCC',
            ...     'C1=CC=CC=C1'
            ... ]
            >>> mols = [Mol.from_smiles(x, make_3D=True) for x in smi]
            >>> vp = VoxelParams(
            ...     resolution=0.75,
            ...     width=24,
            ...     atom_featurizer=AtomicNumFeaturizer([1,6,7,8,16])
            ... )
            >>> t = torch.zeros(vp.tensor_size(batch=4))
            >>> for i in range(len(mols)):
            ...     mols[i].voxelize_into(t, i, vp, cpu=True)
            >>> print(t.shape)
            torch.Size([4, 5, 24, 24, 24])

        Args:
            tensor (torch.Tensor): A 5-D PyTorch Tensor that will receive atomic density information. The tensor
                must have shape BxNxWxWxW. B = batch size, N = number of atom layers, W = width.
            batch_idx (int): An integer specifying which index to write density into. (0 <= batch_idx < B)
            params (VoxelParams): A VoxelParams object specifying how to perform voxelation.
            cpu (bool): If True, will force computation to run on the CPU.
            layer_offset (int): An optional integer specifying a start layer for voxelation.
            center (np.ndarray): A size 3 array containing the (x,y,z) coordinate of the grid center. If not specified,
                will use the center of the molecule.
            rot (np.ndarray): A size 4 quaternion in form (x,y,z,w) describing a grid rotation.
        """
        grid = numba_ptr(tensor, cpu=cpu)

        mol_gridify(
            grid=grid,
            atom_coords=self.coords,
            atom_mask=params.atom_featurizer.featurize_mol(self),
            layer_offset=layer_offset,
            batch_idx=batch_idx,
            width=params.width,
            res=params.resolution,
            center=(center if center is not None else self.center),
            rot=rot,
            point_radius=params.point_radius,
            point_type=params.point_type.value,
            acc_type=params.acc_type.value,
            cpu=cpu,
        )

    def fingerprint(self, fp_type: str, size: int) -> "np.array":
        return fingerprint_for(self.rdmol, fp_type, size)


class MolDataset(Dataset):
    """
    Abstract interface for a MolDataset object.

    Subclasses should implement __len__ and __getitem__ to support enumeration
    over the data.
    """

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> "Mol":
        raise NotImplementedError()
