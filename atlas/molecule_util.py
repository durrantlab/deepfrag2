from typing import List, Tuple, Set, Dict

import numpy as np
from openbabel import pybel
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


from atlas.draw import DrawContext


# Shorthand type definitions.
MolEdge = Tuple[int, int]


def _pybel_bond_index(mol: pybel.Molecule) -> np.ndarray:
    bonds = [mol.OBMol.GetBondById(i) for i in range(mol.OBMol.NumBonds())]
    edges = []

    for b in bonds:
        # Atom indexes start at 1.
        start = b.GetBeginAtomIdx() - 1
        end = b.GetEndAtomIdx() - 1
        edges.append((start, end))

    return np.array(edges).transpose()


def _pybel_bond_quality(mol: pybel.Molecule) -> Dict[MolEdge, int]:
    bonds = [mol.OBMol.GetBondById(i) for i in range(mol.OBMol.NumBonds())]
    quality = {}

    for b in bonds:
        # Atom indexes start at 1.
        start = b.GetBeginAtomIdx() - 1
        end = b.GetEndAtomIdx() - 1

        if end < start:
            start, end = end, start

        quality[(start, end)] = b.GetBondOrder()

    return quality


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
    ATOMS = [
        6,
        8,
        7,
        16,
        9,
        15,
        17,
        35,
        5,
        53,
        34,
        26,
        27,
        44,
        14,
        29,
        33,
        45,
        4,
        30,
        77,
        12,
        75,
        23,
        51,
        76,
        80,
    ]

    @staticmethod
    def featurize(mol: pybel.Molecule) -> np.ndarray:
        atomnum = [x.atomicnum for x in mol.atoms]

        # idx 0 represents "other"
        atom_types = np.zeros((len(atomnum), len(SimpleAtomFeaturizer.ATOMS) + 1))
        for i in range(len(atomnum)):
            if atomnum[i] in SimpleAtomFeaturizer.ATOMS:
                atom_types[i][SimpleAtomFeaturizer.ATOMS.index(atomnum[i]) + 1] = 1
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

                if order == 1:
                    bond_types[i][1] = 1
                elif order == 2:
                    bond_types[i][2] = 1
                elif order == 3:
                    bond_types[i][3] = 1

            if bond.IsInRing():
                bond_types[i][4] = 1

        return bond_types


def _cast_tensor(arr, dtype):
    if type(arr) is torch.Tensor:
        return arr
    else:
        return torch.tensor(arr, dtype=dtype)


class MolGraph(Data):
    """Molecule with 3D coordinates.

    Attributes:
    (N atoms) (B bonds)
    - atom_coords: Nx3 array of (x,y,z) coordinates.
    - atom_types: Nx? array of atom feature vectors.
    - bond_index: 2x(B*2) edge_index array.
    - bond_types: (B*2)x? array of bond feature vectors.

    Metadata:
    - bond_quality: MolEdge -> int
    """

    def __init__(self, atom_coords, atom_types, bond_index, bond_types, bond_quality):
        self.atom_coords = _cast_tensor(atom_coords, dtype=torch.float)
        self.atom_types = _cast_tensor(atom_types, dtype=torch.float)
        self.bond_index = _cast_tensor(bond_index, dtype=torch.long)
        self.bond_types = _cast_tensor(bond_types, dtype=torch.float)
        self.bond_quality = bond_quality

        # MolGraph objects may contain a smiles string if initialized with
        # MolGraph.from_smiles().
        self.smiles: str = None

        # An optional dict with metadata information about the origin of the
        # MolGraph object. E.g. may contain a "ZINC" indentifier.
        self.meta: dict = {}

    def clone(self) -> "MolGraph":
        g = MolGraph(
            self.atom_coords.clone(),
            self.atom_types.clone(),
            self.bond_index.clone(),
            self.bond_types.clone(),
            self.bond_quality.copy(),
        )
        g.smiles = self.smiles
        g.meta = {k: self.meta[k] for k in self.meta}
        return g

    def __repr__(self):
        return (
            f"MolGraph(atom_coords={self.atom_coords.shape}, "
            f"atom_types={self.atom_types.shape}, "
            f"bond_index={self.bond_index.shape}, "
            f"bond_types={self.bond_types.shape})"
        )

    @staticmethod
    def from_pybel(
        mol: pybel.Molecule,
        af: AtomFeaturizer = SimpleAtomFeaturizer,
        bf: BondFeaturizer = SimpleBondFeaturizer,
    ) -> "MolGraph":
        atom_coords = np.array([x.coords for x in mol.atoms])
        atom_types = af.featurize(mol)
        bond_index = _pybel_bond_index(mol)
        bond_types = bf.featurize(mol)
        bond_quality = _pybel_bond_quality(mol)

        return MolGraph(atom_coords, atom_types, bond_index, bond_types, bond_quality)

    @staticmethod
    def from_sdf(sdf_path: str) -> "MolGraph":
        raise NotImplementedError()

    @staticmethod
    def from_smiles(
        smiles: str,
        af: AtomFeaturizer = SimpleAtomFeaturizer,
        bf: BondFeaturizer = SimpleBondFeaturizer,
        make_3D: bool = True,
    ) -> "MolGraph":
        """Create a MolGraph from a smiles string. A reasonable conformation
        is automatically inferred."""

        mol = pybel.readstring("smi", smiles)
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
            (self.bond_index, torch.flip(self.bond_index, (0,))), 1
        )

        self.bond_types = torch.cat((self.bond_types, self.bond_types), 0)

    def _make_fake_sdf(self) -> str:
        """Generate a fake carbon skeleton SDF for visualization of abstract
        molecular topologies."""
        edges = self._collect_edges()

        sdf = "\n\n\n"
        sdf += f"{len(self.atom_coords):3d}{len(edges):3d}  0  0  0  0  0  0  0  0999 V2000\n"

        for i in range(len(self.atom_coords)):
            x, y, z = self.atom_coords[i]
            sdf += (
                f"{x:10.4f}{y:10.4f}{z:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            )

        for a, b in edges:
            sdf += f"{a+1:3d}{b+1:3d}  1  0  0  0  0\n"

        sdf += "M  END\n$$$$\n"
        return sdf

    def to_sdf(self) -> str:
        """Convert the molecule to sdf using atom_coords."""
        if self.smiles is None:
            return self._make_fake_sdf()
        else:
            mol = pybel.readstring("smi", self.smiles)
            for i in range(len(mol.atoms)):
                mol.atoms[i].OBAtom.SetVector(*[float(x) for x in self.atom_coords[i]])
            return mol.write("sdf")

    def view(self, highlight=[], **kwargs) -> "py3DMol.view":
        """Render the molecule with py3DMol (for use in jupyter).

        Args:
        - highlight: If a list, this is interpreted as a list of atom indexes
            to highlight. If a dict, this is interpreted as an atom index to
            feature mapping. If the feature is a scalar, it is interpreted as
            the sphere radius, if it is a dictionary, the values are passed
            directly to DrawContext.draw_sphere. For example, you can specify:

            mol.view(highlight={
                0: {
                    'color': 'red',
                    'radius': 0.8,
                    'opacity': 0.5
                }, ...
            })
        """
        draw = DrawContext(**kwargs)
        draw.draw_mol(self)

        if type(highlight) is list:
            # Binary atom selection.
            for idx in highlight:
                center = tuple([float(x) for x in self.atom_coords[idx].cpu().detach()])
                draw.draw_sphere(center, opacity=0.8, radius=0.6)
        else:
            # Feature mapping.
            for idx in highlight:
                center = tuple([float(x) for x in self.atom_coords[idx].cpu().detach()])
                ft = highlight[idx]

                if type(ft) in [int, float]:
                    draw.draw_sphere(center, opacity=0.8, radius=ft)
                else:
                    draw.draw_sphere(center, **ft)

        return draw.render()

    def _collect_edges(self) -> List[MolEdge]:
        edges = [
            tuple(sorted(x))
            for x in self.bond_index.transpose(0, 1).cpu().detach().numpy()
        ]
        edges = list(set(edges))  # dedup
        return edges

    def random_construction(self) -> List[MolEdge]:
        """Generate a random permutation of edges that reconstructs the
        molecular graph.

        This is used as a data pre-processing step for training generative
        graph models.

        This method picks a random starting index and then performs a breadth-
        first search over the atoms until the whole graph is covered. It
        returns a list of MolEdge (Tuple[int,int]) describing the intermediate
        graph transformation steps.

        In each new edge, the first element is the "branching" point and is
        guaranteed to exist in the current subgraph (except for the first edge).

        For example, given the graph:

        0-1-2-4
          |/
          3

        A valid bfs may produce:
        [
            (0,1),
            (1,3),
            (1,2),
            (2,4),
            (3,2)
        ]
        """
        edges = self._collect_edges()

        transformations = []

        subgraph = {np.random.choice(len(self.atom_types))}
        seen_edges = set()

        for i in range(len(edges)):
            valid = [
                x
                for x in edges
                if (x[0] in subgraph or x[1] in subgraph) and not x in seen_edges
            ]

            a, b = valid[np.random.choice(len(valid))]

            # Ensure the first element is contained in the current subgraph.
            if not a in subgraph:
                transformations.append((b, a))
            elif not b in subgraph:
                transformations.append((a, b))
            else:
                transformations.append((a, b) if np.random.choice(2) else (b, a))

            subgraph |= {a, b}
            seen_edges.add((a, b))

        return transformations

    def subgraph(self, edges: List[MolEdge], atoms: Set[int] = set()) -> "MolGraph":
        """Sample a new subgraph from a list of edges. Gradients are preserved
        for atom_types and bond_types.

        The returned MolGraph object has the following meta parameters:
        - "atom_mapping": Dict[int,int] A mapping from the parent molecule atom
            indexes to the subgraph atom indexes.
        """

        selected_atoms = set(atoms)
        for a, b in edges:
            selected_atoms |= {a, b}
        selected_atoms = sorted(list(selected_atoms))

        atom_coords = self.atom_coords[selected_atoms]

        # Preserve computation graph so gradients work.
        atom_types = torch.zeros(
            (len(selected_atoms), self.atom_types.shape[1]),
            dtype=torch.float,
            device=self.atom_types.device,
        )
        for i in range(len(selected_atoms)):
            atom_types[i] = self.atom_types[i]

        selected_bonds = set()
        for i in range(self.bond_index.shape[1]):
            a, b = self.bond_index[:, i].cpu().numpy()
            if (a, b) in edges or (b, a) in edges:
                selected_bonds.add(i)
        selected_bonds = list(selected_bonds)

        bond_index = self.bond_index[:, selected_bonds]
        atom_mapping = {selected_atoms[i]: i for i in range(len(selected_atoms))}

        bv = bond_index.view(-1)
        for i in range(len(bv)):
            bv[i] = atom_mapping[int(bv[i])]

        bond_types = torch.zeros(
            (len(selected_bonds), self.bond_types.shape[1]),
            dtype=torch.float,
            device=self.atom_types.device,
        )
        for i in range(len(selected_bonds)):
            bond_types[i] = self.bond_types[i]

        sub = MolGraph(
            atom_coords,
            atom_types,
            bond_index,
            bond_types,
            {self.bond_quality[k] for k in edges},
        )
        sub.meta["atom_mapping"] = atom_mapping

        return sub

    def add_atom(self, coord, link: int, copy: bool = True) -> "MolGraph":
        """Add an atom and bond to the graph.

        This method duplicates the current graph and appends a new entry to
        atom_coords and atom_types. It also adds two entries to bond_index and
        bond_types to create a bidirectional edge.

        By default, atom_types and bond_types are filled with zero vectors.

        Args:
        - coord: list-like (x,y,z) coordinate of the new atom.
        - link: index of the atom to create a bond with.
        - copy: If True (default), create a copy of the graph, otherwise edit
            the current graph in-place.
        """
        mol = self
        if copy:
            mol = self.clone()

        # SMILES is no longer accurate.
        mol.smiles = None

        mol.atom_coords = torch.cat(
            [
                mol.atom_coords,
                _cast_tensor(coord, dtype=torch.float)
                .unsqueeze(0)
                .to(mol.atom_coords.device),
            ],
            axis=0,
        )

        mol.atom_types = torch.cat(
            [
                mol.atom_types,
                torch.zeros((1, mol.atom_types.shape[1])).to(mol.atom_types.device),
            ],
            axis=0,
        )

        new_idx = len(mol.atom_types) - 1

        mol.bond_index = torch.cat(
            [
                mol.bond_index,
                torch.tensor([[new_idx, link], [link, new_idx]]).to(
                    mol.bond_index.device
                ),
            ],
            axis=1,
        )

        mol.bond_types = torch.cat(
            [
                mol.bond_types,
                torch.zeros((2, mol.bond_types.shape[1])).to(mol.bond_types.device),
            ],
            axis=0,
        )

        return mol

    def add_bond(self, a: int, b: int, copy: bool = True) -> "MolGraph":
        """Add a new edge to the graph between index "a" and "b".

        Args:
        - a: Index of the first atom.
        - b: Index of the second atom.
        - copy: If True (default), create a copy of the graph, otherwise edit
            the current graph in-place.
        """
        mol = self
        if copy:
            mol = self.clone()

        # SMILES is no longer accurate.
        mol.smiles = None

        mol.bond_index = torch.cat(
            [mol.bond_index, torch.tensor([[a, b], [b, a]]).to(mol.bond_index.device)],
            axis=1,
        )

        mol.bond_types = torch.cat(
            [
                mol.bond_types,
                torch.zeros((2, mol.bond_types.shape[1])).to(mol.bond_types.device),
            ],
            axis=0,
        )

        return mol

    def pairwise_dist(self) -> torch.Tensor:
        """Returns the pairwise distance between all atoms."""
        dist = self.atom_coords.view(-1, 1, 3) - self.atom_coords.view(1, -1, 3)
        dist = torch.sqrt(torch.sum(dist ** 2, axis=2))
        return dist

    def _trace_cut(self, atom: int, edges: MolEdge) -> Tuple[Set[MolEdge], Set[int]]:
        seen = set()
        atoms = [atom]
        trace_edges = set()

        while len(atoms) > 0:
            a = atoms.pop(0)
            seen.add(a)

            for x, y in edges:
                if a == x and not y in seen and not y in atoms:
                    atoms.append(y)
                    trace_edges.add((x, y))
                elif a == y and not x in seen and not x in atoms:
                    atoms.append(x)
                    trace_edges.add((x, y))

        return (trace_edges, seen)

    def fragments(self, only_single_bonds=True) -> List[Tuple["MolGraph", "MolGraph"]]:
        """Iterate over edges and perform a cut. Returns a list of
        (parent, fragment) tuples."""
        frags = []

        edges = self._collect_edges()

        for i in range(len(edges)):
            if only_single_bonds and self.bond_quality[edges[i]] != 1:
                continue

            frag1 = self._trace_cut(
                edges[i][0], [edges[k] for k in range(len(edges)) if k != i]
            )

            frag2 = self._trace_cut(
                edges[i][1], [edges[k] for k in range(len(edges)) if k != i]
            )

            if frag1[1] != frag2[1]:

                if len(frag1[1]) < len(frag2[1]):
                    frag1, frag2 = frag2, frag1

                frags.append((self.subgraph(*frag1), self.subgraph(*frag2)))

        return frags


class MolGraphProvider(Dataset):
    """Abstract interface for a MolGraphProvider object.

    Subclasses should implement __len__ and __getitem__ to support enumeration
    over the data.
    """

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx) -> MolGraph:
        raise NotImplementedError()
