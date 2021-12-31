from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union, Tuple, Set, Optional, Any, Callable
import os
import json
import numpy as np
import prody
from torch.utils.data import Dataset
import numpy

from ...core.mol import Mol


def _fix_moad_smiles(smi):
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
                        lig_atoms, _fix_moad_smiles(lig.smiles), sanitize=True
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


def _split_seq(seq, p):
    l = list(seq)
    sz = len(l)
    np.random.shuffle(l)

    return l[: int(sz * p)], l[int(sz * p) :]


def _flatten(seq):
    a = []
    for s in seq:
        a += s
    return a


def _div2(seq):
    l = list(seq)
    sz = len(l)

    np.random.shuffle(l)

    return (set(l[: sz // 2]), set(l[sz // 2 :]))


def _div3(seq):
    l = list(seq)
    sz = len(l)

    np.random.shuffle(l)

    v = sz // 3
    return (set(l[:v]), set(l[v : v * 2]), set(l[v * 2 :]))


class MOADInterface(object):
    """
    Base class for interacting with Binding MOAD data. Initialize by passing the path to
    "every.csv" and the path to a folder containing structure files (can be nested).

    Args:
        metadata: Path to the metadata "every.csv" file.
        structures: Path to a folder container structure files.
    """

    classes: List["MOAD_class"]
    _lookup: Dict["str", "MOAD_target"] = field(default_factory=dict)
    _all_targets: List["str"] = field(default_factory=list)

    def __init__(self, metadata: Union[str, Path], structures: Union[str, Path]):
        self.classes = MOADInterface._load_classes(metadata)
        self._lookup = {}
        self._all_targets = []

        self._init_lookup()
        self._resolve_paths(structures)

    def _init_lookup(self):
        for c in self.classes:
            for f in c.families:
                for t in f.targets:
                    self._lookup[t.pdb_id.lower()] = t

        self._all_targets = [k for k in self._lookup]

    @property
    def targets(self) -> List["str"]:
        return self._all_targets

    def __getitem__(self, key: str) -> "MOAD_target":
        """
        Fetch a specific target by PDB ID.

        Args:
            key (str): A PDB ID (case-insensitive).

        Returns:
            MOAD_target: a MOAD_target object if found.
        """
        assert type(key) is str, f"PDB ID must be a str (got {type(key)})"
        k = key.lower()
        assert k in self._lookup, f'Target "{k}" not found.'
        return self._lookup[k]

    @staticmethod
    def _load_classes(path):
        with open(path, "r") as f:
            dat = f.read().strip().split("\n")

        classes = []
        curr_class = None
        curr_family = None
        curr_target = None

        for line in dat:
            parts = line.split(",")

            if parts[0] != "":  # 1: Protein Class
                if curr_class is not None:
                    classes.append(curr_class)
                curr_class = MOAD_class(ec_num=parts[0], families=[])
            elif parts[1] != "":  # 2: Protein Family
                if curr_target is not None:
                    curr_family.targets.append(curr_target)
                if curr_family is not None:
                    curr_class.families.append(curr_family)
                curr_family = MOAD_family(rep_pdb_id=parts[2], targets=[])
                curr_target = MOAD_target(pdb_id=parts[2], ligands=[])
            elif parts[2] != "":  # 3: Protein target
                if curr_target is not None:
                    curr_family.targets.append(curr_target)
                curr_target = MOAD_target(pdb_id=parts[2], ligands=[])
            elif parts[3] != "":  # 4: Ligand
                curr_target.ligands.append(
                    MOAD_ligand(
                        name=parts[3],
                        validity=parts[4],
                        affinity_measure=parts[5],
                        affinity_value=parts[7],
                        affinity_unit=parts[8],
                        smiles=parts[9],
                    )
                )

        if curr_target is not None:
            curr_family.targets.append(curr_target)
        if curr_family is not None:
            curr_class.families.append(curr_family)
        if curr_class is not None:
            classes.append(curr_class)

        return classes

    def _resolve_paths(self, path: Union[str, Path]):
        path = Path(path)

        files = {}
        for f in path.glob("./**/*.bio*"):
            pdbid = f.stem
            if not pdbid in files:
                files[pdbid] = []
            files[pdbid].append(f)

        for c in self.classes:
            for f in c.families:
                for t in f.targets:
                    k = t.pdb_id.lower()
                    if k in files:
                        t.files = sorted(files[k])
                    else:
                        # No structures for this pdb id!
                        pass

    def _smiles_for(self, targets: List[str]) -> Set[str]:
        """Return all the SMILES strings contained in the selected targets."""
        smiles = set()

        for target in targets:
            for ligand in self[target].ligands:
                smi = ligand.smiles
                if ligand.is_valid and smi not in ["n/a", "NULL"]:
                    smiles.add(smi)

        return smiles

    def compute_split(
        self,
        seed: int = 0,
        p_train: float = 0.6,
        p_val: float = 0.5,
        prevent_smiles_overlap: bool = True,
    ) -> Tuple["MOAD_split", "MOAD_split", "MOAD_split"]:
        """Compute a TRAIN/VAL/TEST split.

        Targets are first assigned to a TRAIN set with `p_train` probability. The remaining targets are assigned to a VAL set with
        `p_val` probability. The unused targets are assigned to the TEST set.

        Args:
            seed (int, optional): If set to a nonzero number, compute_split will always return the same split.
            p_train (float, optional): Percentage of targets to use in the TRAIN set.
            p_val (float, optional): Percentage of (non-train) targets to use in the VAL set.

        Returns:
            Tuple[MOAD_split, MOAD_split, MOAD_split]: train/val/test sets
        """

        if seed != 0:
            np.random.seed(seed)

        families: List[List[str]] = []
        import pdb; pdb.set_trace()
        for c in self.classes[:50]:
            for f in c.families:
                families.append([x.pdb_id for x in f.targets])

        train_f, other_f = _split_seq(families, p_train)
        val_f, test_f = _split_seq(other_f, p_val)

        train_ids = _flatten(train_f)
        val_ids = _flatten(val_f)
        test_ids = _flatten(test_f)

        train_smi = self._smiles_for(train_ids)
        val_smi = self._smiles_for(val_ids)
        test_smi = self._smiles_for(test_ids)

        if prevent_smiles_overlap:
            # Reassign overlapping SMILES.
            a_train, a_val = _div2((train_smi & val_smi) - test_smi)
            b_val, b_test = _div2((val_smi & test_smi) - train_smi)
            c_train, c_test = _div2((train_smi & test_smi) - val_smi)
            d_train, d_val, d_test = _div3((train_smi & val_smi & test_smi))

            train_smi = (train_smi - (val_smi | test_smi)) | a_train | c_train | d_train
            val_smi = (val_smi - (train_smi | test_smi)) | a_val | b_val | d_val
            test_smi = (test_smi - (train_smi | val_smi)) | b_test | c_test | d_test

        # Save spit and seed to json in working directory if running in docker
        # container.
        if os.path.exists("/working/"):
            split_inf = {
                "seed": seed,
                "train": train_ids,
                "val": val_ids,
                "test": test_ids
            }
            json.dump(split_inf, open("/working/moad_split.json", "w"), indent=4)

        return (
            MOAD_split(name="TRAIN", targets=train_ids, smiles=train_smi),
            MOAD_split(name="VAL", targets=val_ids, smiles=val_smi),
            MOAD_split(name="TEST", targets=test_ids, smiles=test_smi),
        )

    def full_split(self) -> MOAD_split:
        """Returns a split containing all targets and smiles strings."""
        return MOAD_split(
            name="Full",
            targets=self.targets,
            smiles=list(self._smiles_for(self.targets)),
        )


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
