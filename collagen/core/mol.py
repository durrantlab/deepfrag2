from dataclasses import dataclass
from io import StringIO
from typing import List, Tuple, Iterator, Any, Dict, Optional
import warnings

import numpy as np
import prody
import rdkit.Chem.AllChem as Chem
from rdkit.Chem.Descriptors import ExactMolWt
import torch
from torch.utils.data import Dataset

from .fingerprints import fingerprint_for
from .voxelizer import numba_ptr, mol_gridify
from .types import AnyAtom
from ..draw import MolView


class Mol(object):
    meta: Dict[str, Any]

    _KW_MOL_NAME = "name"

    def __init__(self, meta: dict = None):
        if meta is None:
            self.meta = {}
        else:
            self.meta = meta

    def __repr__(self):
        _cls = type(self).__name__
        if Mol._KW_MOL_NAME in self.meta:
            return f'{_cls}("{self.meta[Mol._KW_MOL_NAME]}")'
        else:
            return f"{_cls}()"

    @staticmethod
    def from_smiles(
        smiles: str, sanitize: bool = False, make_3D: bool = False
    ) -> "BackedMol":
        """Construct a Mol from a SMILES string.

        Notes:
            By default, the molecule does not have 3D coordinate information. Set ``make_3D=True`` to generate a 3D embedding with RDKit.

        Args:
            smiles (str): A SMILES string.
            sanitize (bool, optional): If True, attempt to sanitize the internal RDKit molecule.
            make_3D (bool, optional): If True, generate 3D coordinates.

        Returns:
            collagen.core.mol.BackedMol: A new Mol object.

        Examples:
            Load aspirin from a SMILES string:

            >>> Mol.from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')
            Mol(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        """
        rdmol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        rdmol.UpdatePropertyCache()
        if make_3D:
            Chem.EmbedMolecule(rdmol)

        return BackedMol(rdmol=rdmol)

    @staticmethod
    def from_prody(
        atoms: "prody.atomic.atomgroup.AtomGroup",
        template: str = "",
        sanitize: bool = False,
    ) -> "BackedMol":
        """Construct a Mol from a ProDy AtomGroup.

        Args:
            atoms (prody.atomic.atomgroup.AtomGroup): A ProDy atom group.
            template (str, optional): An optional SMILES string used as a template to assign bond orders.
            sanitize (bool, optional): If True, attempt to sanitize the internal RDKit molecule.

        Returns:
            collagen.core.mol.BackedMol: A new Mol object.

        Examples:
            Extract the aspirin ligand from the 1OXR PDB structure:

            >>> g = prody.parsePDB(prody.fetchPDB('1OXR'))
            >>> g = g.select('resname AIN')
            >>> m = Mol.from_prody(g, template='CC(=O)Oc1ccccc1C(=O)O')
            >>> print(m.coords)
            [[13.907 16.13   0.624]
            [13.254 15.778  1.723]
            [13.911 15.759  2.749]
            [11.83  15.316  1.664]
            [11.114 15.381  0.456]
            [ 9.774 15.001  0.429]
            [ 9.12  14.601  1.58 ]
            [ 9.752 14.568  2.802]
            [11.088 14.922  2.923]
            [11.823 14.906  4.09 ]
            [12.477 13.77   4.769]
            [12.686 13.87   5.971]
            [12.89  12.509  4.056]]
        """
        pdb = StringIO()
        prody.writePDBStream(pdb, atoms)

        rdmol = Chem.MolFromPDBBlock(pdb.getvalue(), sanitize=sanitize)

        if template != "":
            ref_mol = Chem.MolFromSmiles(template, sanitize=False)
            # Remove stereochemistry and explicit hydrogens so AssignBondOrdersFromTemplate works.
            Chem.RemoveStereochemistry(ref_mol)
            ref_mol = Chem.RemoveAllHs(ref_mol)
            rdmol.UpdatePropertyCache()
            rdmol = Chem.AssignBondOrdersFromTemplate(ref_mol, rdmol)

        rdmol.UpdatePropertyCache(strict=False)
        return BackedMol(rdmol=rdmol)

    @staticmethod
    def from_rdkit(rdmol: "rdkit.Chem.rdchem.Mol", strict: bool = True) -> "BackedMol":
        """Construct a Mol from an RDKit Mol.

        Args:
            rdmol (rdkit.Chem.rdchem.Mol): An existing RDKit Mol.

        Returns:
            collagen.core.mol.BackedMol: A new Mol object.
        """
        rdmol.UpdatePropertyCache(strict=strict)
        return BackedMol(rdmol=rdmol)

    def sdf(self) -> str:
        raise NotImplementedError()

    def pdb(self) -> str:
        raise NotImplementedError()

    def smiles(self, isomeric: bool = False) -> str:
        """
        Compute a SMILES string for this Mol.

        Args:
            isomeric (bool, optional): True if this string should be isomeric.

        Returns:
            str: A SMILES string.
        """
        raise NotImplementedError()

    @property
    def coords(self) -> "numpy.ndarray":
        """Atomic coordinates as a numpy array."""
        return NotImplementedError()

    @property
    def center(self) -> "numpy.ndarray":
        """The average atomic coordinate of this Mol."""
        return np.mean(self.coords, axis=0)

    @property
    def atoms(self) -> List[AnyAtom]:
        """Atoms in this Mol."""
        return NotImplementedError()

    @property
    def num_atoms(self) -> int:
        """Number of atoms in this Mol."""
        return len(self.atoms)

    @property
    def num_heavy_atoms(self) -> int:
        """Number of heavy atoms in this Mol."""
        raise NotImplementedError()

    @property
    def connectors(self) -> List["numpy.ndarray"]:
        """Return a list of connector atom coordinates."""
        raise NotImplementedError()

    @property
    def mass(self) -> float:
        """Mass of this Mol in daltons."""
        raise NotImplementedError()

    def voxelize(
        self,
        params: "VoxelParams",
        cpu: bool = False,
        center: "numpy.ndarray" = None,
        rot: "numpy.ndarray" = np.array([0, 0, 0, 1]),
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
            center: (numpy.ndarray): Optional, if set, center the grid on this 3D coordinate.
            rot: (numpy.ndarray): A size 4 array describing a quaternion rotation for the grid.
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
        center: "numpy.ndarray" = None,
        rot: "numpy.ndarray" = np.array([0, 0, 0, 1]),
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
            center (numpy.ndarray): A size 3 array containing the (x,y,z) coordinate of the grid center. If not specified,
                will use the center of the molecule.
            rot (numpy.ndarray): A size 4 quaternion in form (x,y,z,w) describing a grid rotation.
        """
        grid = numba_ptr(tensor, cpu=cpu)
        atom_mask, atom_radii = params.atom_featurizer.featurize_mol(self)
        mol_gridify(
            grid=grid,
            atom_coords=self.coords,
            atom_mask=atom_mask,
            atom_radii=atom_radii,
            layer_offset=layer_offset,
            batch_idx=batch_idx,
            width=params.width,
            res=params.resolution,
            center=(center if center is not None else self.center),
            rot=rot,
            atom_scale=params.atom_scale,
            atom_shape=params.atom_shape.value,
            acc_type=params.acc_type.value,
            cpu=cpu,
        )

    def voxelize_delayed(
        self,
        params: "VoxelParams",
        center: "numpy.ndarray" = None,
        rot: "numpy.ndarray" = np.array([0, 0, 0, 1]),
    ) -> "DelayedMolVoxel":
        """
        Pre-compute voxelation parameters without actually invoking ``voxelize``.

        Args:
            params (VoxelParams): A VoxelParams object specifying how to perform voxelation.
            center (numpy.ndarray): A size 3 array containing the (x,y,z) coordinate of the grid center. If not specified,
                will use the center of the molecule.
            rot (numpy.ndarray): A size 4 quaternion in form (x,y,z,w) describing a grid rotation.

        Returns:
            DelayedMolVoxel: An ephemeral, minimal Mol object with pre-computed voxelation arguments.
        """
        params.validate()
        atom_mask, atom_radii = params.atom_featurizer.featurize_mol(self)
        return DelayedMolVoxel(
            atom_coords=self.coords,
            atom_mask=atom_mask,
            atom_radii=atom_radii,
            width=params.width,
            res=params.resolution,
            center=(center if center is not None else self.center),
            rot=rot,
            atom_scale=params.atom_scale,
            atom_shape=params.atom_shape.value,
            acc_type=params.acc_type.value,
        )

    def stick(self, **kwargs) -> "py3DMol.view":
        """Render the molecule with py3DMol (for use in jupyter)."""
        draw = MolView(**kwargs)
        draw.add_stick(self)
        return draw.render()

    def sphere(self, **kwargs) -> "py3DMol.view":
        """Render the molecule with py3DMol (for use in jupyter)."""
        draw = MolView(**kwargs)
        draw.add_sphere(self)
        return draw.render()

    def cartoon(self, **kwargs) -> "py3DMol.view":
        """Render the molecule with py3DMol (for use in jupyter)."""
        draw = MolView(**kwargs)
        draw.add_cartoon(self)
        return draw.render()


class DelayedMolVoxel(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def voxelize_into(
        self,
        tensor: "torch.Tensor",
        batch_idx: int,
        layer_offset: int = 0,
        cpu: bool = False,
    ):
        grid = numba_ptr(tensor, cpu=cpu)

        mol_gridify(
            grid=grid,
            batch_idx=batch_idx,
            layer_offset=layer_offset,
            cpu=cpu,
            **self.kwargs,
        )


class BackedMol(Mol):
    """A BackedMol is a thin wrapper over an RDKit molecule."""

    def __init__(self, rdmol: "rdkit.Chem.rdchem.Mol", meta: dict = None):
        """Initialize a new BackedMol with an existing RDMol."""
        super(BackedMol, self).__init__(meta=meta)
        self.rdmol = rdmol

        if self.rdmol.GetNumConformers() == 0:
            warnings.warn("Internal rdmol has no conformers")

    def __repr__(self):
        _cls = type(self).__name__
        if Mol._KW_MOL_NAME in self.meta:
            return f'{_cls}("{self.meta[Mol._KW_MOL_NAME]}")'
        else:
            return f'{_cls}(smiles="{self.smiles()}")'

    def _ensure_structure(self):
        assert (
            self.rdmol.GetNumConformers() > 0
        ), "Error: RDMol has no coordinate information."

    def sdf(self) -> str:
        """Convert to SDF format."""
        self._ensure_structure()
        s = StringIO()
        w = Chem.SDWriter(s)
        w.write(self.rdmol)
        w.close()
        return s.getvalue()

    def pdb(self) -> str:
        self._ensure_structure()
        return Chem.MolToPDBBlock(self.rdmol)

    def smiles(self, isomeric: bool = False) -> str:
        """Convert the internal rdmol to a SMILES string.

        Note:
            This version returns a non-isomeric SMILES.
        """
        return Chem.MolToSmiles(self.rdmol, isomericSmiles=isomeric)

    @property
    def coords(self) -> "numpy.ndarray":
        """Return atomic coordinates as a numpy array."""
        self._ensure_structure()
        return self.rdmol.GetConformer().GetPositions()

    @property
    def connectors(self) -> List["numpy.ndarray"]:
        self._ensure_structure()
        return [
            self.coords[atom.GetIdx()]
            for atom in self.atoms
            if atom.GetAtomicNum() == 0
        ]

    @property
    def atoms(self) -> List[AnyAtom]:
        return list(self.rdmol.GetAtoms())

    @property
    def num_heavy_atoms(self) -> int:
        return self.rdmol.GetNumHeavyAtoms()

    @property
    def mass(self) -> float:
        return ExactMolWt(self.rdmol)

    def split_bonds(
        self, only_single_bonds: bool = True, max_frag_size: int = -1
    ) -> List[Tuple["Mol", "Mol"]]:
        """
        Iterate over all bonds in the Mol and try to split into two fragments, returning tuples of produced fragments.
        Each returned tuple is of the form (parent, fragment).

        Args:
            only_single_bonds (bool): If True (default) only cut on single bonds.
            max_frag_size (int): If set, only return fragments smaller or equal to this molecular weight.

        Examples:
            >>> mol = Mol.from_smiles('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O')
            >>> mol.split_bonds()
            [(Mol(smiles="*C(C)CC1=CC=C(C(C)C(=O)O)C=C1"), Mol(smiles="*C")),
            (Mol(smiles="*C(C)CC1=CC=C(C(C)C(=O)O)C=C1"), Mol(smiles="*C")),
            (Mol(smiles="*CC1=CC=C(C(C)C(=O)O)C=C1"), Mol(smiles="*C(C)C")),
            (Mol(smiles="*C1=CC=C(C(C)C(=O)O)C=C1"), Mol(smiles="*CC(C)C")),
            (Mol(smiles="*C1=CC=C(CC(C)C)C=C1"), Mol(smiles="*C(C)C(=O)O")),
            (Mol(smiles="*C(C(=O)O)C1=CC=C(CC(C)C)C=C1"), Mol(smiles="*C")),
            (Mol(smiles="*C(C)C1=CC=C(CC(C)C)C=C1"), Mol(smiles="*C(=O)O")),
            (Mol(smiles="*C(=O)C(C)C1=CC=C(CC(C)C)C=C1"), Mol(smiles="*O"))]
        """
        num_mols = len(Chem.GetMolFrags(self.rdmol, asMols=True, sanitizeFrags=False))
        assert (
            num_mols == 1
        ), f"Error, calling split_bonds() on a Mol with {num_mols} parts."

        pairs = []
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
            if len(fragments) != 2:
                continue

            parent = Mol.from_rdkit(fragments[0])
            frag = Mol.from_rdkit(fragments[1])

            if parent.mass < frag.mass:
                frag, parent = parent, frag

            # Ensure the fragment has at least one heavy atom.
            if frag.num_heavy_atoms == 0:
                continue

            # Filter by atomic mass (if enabled).
            if max_frag_size != -1 and frag.mass > max_frag_size:
                continue

            pairs.append((parent, frag))

        return pairs

    def fingerprint(self, fp_type: str, size: int) -> "numpy.ndarray":
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
