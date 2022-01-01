from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collagen.external.moad.moad_interface import MOADInterface

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Tuple, Optional, Any, Callable
import numpy as np
from tqdm.std import tqdm
from collagen.external.moad.cache import build_moad_cache
import prody
from torch.utils.data import Dataset
import numpy

from ...core.mol import Mol


def fix_moad_smiles(smi):
    return (
        smi.replace("+H3", "H3+")
        .replace("+H2", "H2+")
        .replace("+H", "H+")
        .replace("-H", "H-")
        .replace("Al-11H0", "Al-")  # Strange smiles in pdb 2WZC
    )


@dataclass
class MOAD_class(object):
    ec_num: str
    families: List["MOAD_family"]


@dataclass
class MOAD_family(object):
    rep_pdb_id: str
    targets: List["MOAD_target"]


@dataclass
class MOAD_target(object):
    pdb_id: str
    ligands: List["MOAD_ligand"]
    files: List[Path] = field(default_factory=list)

    def __len__(self) -> int:
        """Returns the number of on-disk structures."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple["Mol", "Mol"]:
        """
        Load the Nth structure for this target.

        Args:
            idx (int): The index of the biological assembly to load.

        Returns a (receptor, ligand) tuple of :class:`atlas.data.mol.Mol` objects.
        """
        f = open(self.files[idx], "r")
        m = prody.parsePDBStream(f)
        f.close()

        ignore_sels = []
        ligands = []
        for lig in self.ligands:
            lig_sel = f"chain {lig.chain} and resnum >= {lig.resnum} and resnum < {lig.resnum + lig.reslength}"

            if lig.validity != "Part of Protein":
                ignore_sels.append(lig_sel)

            if lig.is_valid:
                lig_atoms = m.select(lig_sel)

                # Ligand may not be present in this biological assembly.
                if lig_atoms is None:
                    continue

                try:
                    lig_mol = Mol.from_prody(
                        lig_atoms, fix_moad_smiles(lig.smiles), sanitize=True
                    )
                    lig_mol.meta["name"] = lig.name
                    lig_mol.meta["moad_ligand"] = lig
                except Exception:
                    # Ligand SMILES did not match actual geometry.
                    continue

                ligands.append(lig_mol)

        if len(ignore_sels) > 0:
            rec_sel = "not water and not (%s)" % " or ".join(
                f"({x})" for x in ignore_sels
            )
        else:
            rec_sel = "not water"
        receptor = Mol.from_prody(m.select(rec_sel))
        receptor.meta["name"] = f"Receptor {self.pdb_id.lower()}"

        return receptor, ligands


@dataclass
class MOAD_ligand(object):
    name: str
    validity: str
    affinity_measure: str
    affinity_value: str
    affinity_unit: str
    smiles: str

    @property
    def chain(self) -> str:
        return self.name.split(":")[1]

    @property
    def resnum(self) -> int:
        return int(self.name.split(":")[2])

    @property
    def reslength(self) -> int:
        return len(self.name.split(":")[0].split(" "))

    @property
    def is_valid(self) -> bool:
        return self.validity == "valid"


@dataclass
class MOAD_split(object):
    name: str
    targets: List[str]
    smiles: List[str]


MOAD_REF = None


def _unit_rand(thresh):
    u = np.random.uniform(size=3)
    u = (u * 2) - 1
    u /= np.sqrt(np.sum(u * u))
    u *= np.random.rand() * np.sqrt(thresh)
    return u


def _sample_near(coords, thresh):
    idx = np.random.choice(len(coords))
    c = coords[idx]

    offset = _unit_rand(thresh)
    p = c + offset

    return p


def _sample_inside(bmin, bmax, thresh, avoid):
    while True:
        p = np.random.uniform(size=3)
        p = (p * (bmax - bmin)) + bmin

        bad = False
        for i in range(len(avoid)):
            d = np.sum((avoid[i] - p) ** 2)
            if np.sqrt(d) <= thresh + 1e-5:
                bad = True
                break

        if bad:
            continue

        return p


class MOADPocketDataset(Dataset):
    """
    A Dataset that provides (receptor, pos, neg) tuples where pos and neg are points in a binding pocket and outside of a binding pocket respectively.

    Positive samples are genearated by picking a random ligand atom and sampling a random offset. Negative samples are generated
    by randomly sampling a point withing the bounding box of the receptor (plus padding) that is not near any ligand atom.

    Args:
        moad (MOADInterface): An initialized MOADInterface object.
        thresh (float, optional): Threshold to ligand atoms to consider a "binding pocket."
        padding (float, optional): Padding added to receptor bounding box to sample negative examples.
        split (MOAD_split, optional): An optional split to constrain the space of examples.
        transform (Callable[[Mol, numpy.ndarray, numpy.ndarray], Any], optional): An optional transformation function to invoke before returning samples.
            Takes the arguments (receptor, pos, neg).
    """

    def __init__(
        self,
        moad: MOADInterface,
        thresh: float = 3,
        padding: float = 5,
        split: Optional[MOAD_split] = None,
        transform: Optional[
            Callable[[Mol, "numpy.ndarray", "numpy.ndarray"], Any]
        ] = None,
    ):
        self.moad = moad
        self.thresh = thresh
        self.padding = padding
        self.split = split if split is not None else moad.full_split()
        self.transform = transform
        self._index = self._build_index()

    def _build_index(self):
        index = []
        for t in sorted(self.split.targets):
            for n in range(len(self.moad[t])):
                index.append((t, n))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[Mol, "numpy.ndarray", "numpy.ndarray"]:
        target, n = self._index[idx]

        try:
            rec, ligs = self.moad[target][n]
        except:
            return None

        if len(ligs) == 0:
            return None

        lig_coords = np.concatenate([x.coords for x in ligs])
        rec_coords = rec.coords

        pos = _sample_near(lig_coords, self.thresh)

        box_min = np.min(rec_coords, axis=0) - self.padding
        box_max = np.max(rec_coords, axis=0) + self.padding

        neg = _sample_inside(box_min, box_max, self.thresh, avoid=lig_coords)

        out = (rec, pos, neg)

        if self.transform is not None:
            out = self.transform(*out)

        return out


def build_index_and_filter(
    filter_func: Any,
    moad: MOADInterface,
    split: MOAD_split,
    make_dataset_entry_func: Any,
    cache_file: Optional[Union[str, Path]] = None,
    cores: int = 1,
):
    cache_file = Path(cache_file) if cache_file is not None else None

    index = build_moad_cache(cache_file, moad, lig_mass=True, cores=cores)

    internal_index = []
    for pdb_id in tqdm(split.targets, desc="Runtime filters"):
        pdb_id = pdb_id.lower()
        receptor_inf = index[pdb_id]
        for lig_name in receptor_inf.keys():
            lig_inf = receptor_inf[lig_name]

            # Enforce filters. TODO: Distance to receptor, number of heavy
            # atoms, etc.?
            skip = False
            for lig in moad[pdb_id].ligands:
                if lig.name == lig_name:
                    # You've found the ligand.
                    if lig.smiles not in split.smiles:
                        # It is not in the split, so always skip it.
                        skip = True
                        break

                    if not filter_func(lig, lig_inf):
                        # You've found the ligand, but it doesn't pass the
                        # filter.
                        skip = True
                        break

            if skip:
                continue

            internal_index.append(make_dataset_entry_func(pdb_id, lig_name, lig_inf))

    return index, internal_index
