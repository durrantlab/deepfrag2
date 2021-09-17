from dataclasses import dataclass, field
import json
import multiprocessing
from pathlib import Path
from typing import List, Dict, Union, Tuple, Set, Optional, Any, Callable

import numpy as np
import prody
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .mol import Mol, MolDataset


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

    def __getitem__(self, idx: int):
        """"""
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


def div2(seq):
    l = list(seq)
    sz = len(l)

    np.random.shuffle(l)

    return (set(l[: sz // 2]), set(l[sz // 2 :]))


def div3(seq):
    l = list(seq)
    sz = len(l)

    np.random.shuffle(l)

    v = sz // 3
    return (set(l[:v]), set(l[v : v * 2]), set(l[v * 2 :]))


class MOADBase(object):
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
        self.classes = MOADBase.load_classes(metadata)
        self._lookup = {}
        self._all_targets = []

        self._init_lookup()
        self.resolve_paths(structures)

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

        Returns (MOAD_target) a MOAD_target object if found.
        """
        assert type(key) is str, f"PDB ID must be a str (got {type(key)})"
        k = key.lower()
        assert k in self._lookup, f'Target "{k}" not found.'
        return self._lookup[k]

    @staticmethod
    def load_classes(path):
        dat = open(path, "r").read().strip().split("\n")

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

    def resolve_paths(self, path: Union[str, Path]):
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
                        pass
                        # print(f"Warn: no structures for {k}")

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

        Returns (TRAIN, VAL, TEST)
        """

        if seed != 0:
            np.random.seed(seed)

        families: List[List[str]] = []
        for c in self.classes:
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
            a_train, a_val = div2((train_smi & val_smi) - test_smi)
            b_val, b_test = div2((val_smi & test_smi) - train_smi)
            c_train, c_test = div2((train_smi & test_smi) - val_smi)
            d_train, d_val, d_test = div3((train_smi & val_smi & test_smi))

            train_smi = (train_smi - (val_smi | test_smi)) | a_train | c_train | d_train
            val_smi = (val_smi - (train_smi | test_smi)) | a_val | b_val | d_val
            test_smi = (test_smi - (train_smi | val_smi)) | b_test | c_test | d_test

        return (
            MOAD_split(name="TRAIN", targets=train_ids, smiles=train_smi),
            MOAD_split(name="VAL", targets=val_ids, smiles=val_smi),
            MOAD_split(name="TEST", targets=test_ids, smiles=test_smi),
        )

    def full_split(self) -> MOAD_split:
        return MOAD_split(
            name="Full",
            targets=self.targets,
            smiles=list(self._smiles_for(self.targets)),
        )


MOAD_REF = None


def _build_index_target(packed):
    global MOAD_REF

    pdb = packed
    target = MOAD_REF[pdb]
    samples = []

    for i in range(len(target)):
        sample = {}

        try:
            _, ligands = target[i]

            for lig in ligands:
                try:
                    frags = [int(x[1].mass) for x in lig.split_bonds()]
                except:
                    frags = []
                sample[lig.meta["moad_ligand"].name] = frags
        except:
            pass

        samples.append(sample)

    return (pdb, samples)


@dataclass
class MOADFragmentDataset_entry(object):
    pdb_id: str
    sample: int
    ligand: str
    frag_idx: int


class MOADFragmentDataset(Dataset):
    """
    A Dataset that provides (receptor, parent, fragment) tuples by splitting ligands on single bonds.

    Args:
        moad
    """

    moad: MOADBase
    split: MOAD_split
    has_index: bool
    transform: Optional[Callable[[Mol, Mol, Mol], Any]]

    # A cache-able index listing every fragment size for every ligand/target in the dataset.
    #
    # See MOADFragmentDataset._build_index for structure. This index only needs to be updated
    # for new structure files.
    _fragment_index: Optional[dict]

    # The internal listing of every valid fragment example. This index is generated on each
    # run based on the runtime filters: (targets, smiles, fragment_size).
    _internal_index: List[MOADFragmentDataset_entry]

    def __init__(
        self,
        moad: MOADBase,
        cache_file: Optional[Union[str, Path]] = None,
        cache_cores: int = 1,
        split: Optional[MOAD_split] = None,
        transform: Optional[Callable[[Mol, Mol, Mol], Any]] = None,
    ):
        self.moad = moad
        self.split = split if split is not None else moad.full_split()
        self.has_index = False
        self.transform = transform
        self.index(cache_file, cache_cores)

    def _build_index(self, cores: int = 1) -> dict:
        """
        Cache format:

        {
            "pdb_id": [
                {
                    "lig_name": [fmass1, fmass2, ..., fmassN],
                    "lig_name2": [...]
                },
                ...
            ],
            ...
        }
        """
        global MOAD_REF
        MOAD_REF = self.moad

        index = {}
        queue = self.moad.targets

        pbar = tqdm(total=len(queue), desc="Building MOAD cache")
        with multiprocessing.Pool(cores) as p:
            for pdb, item in p.imap_unordered(_build_index_target, queue):
                index[pdb.lower()] = item
                pbar.update(1)

        pbar.close()
        return index

    def index(self, cache_file: Optional[Union[str, Path]] = None, cores: int = 1):
        cache_file = Path(cache_file) if cache_file is not None else None

        if cache_file is not None and cache_file.exists():
            print("Loading MOAD fragments from cache...")
            index = json.load(open(cache_file, "r"))
        else:
            index = self._build_index(cores)
            if cache_file is not None:
                open(cache_file, "w").write(json.dumps(index))

        internal_index = []
        for pdb_id in tqdm(self.split.targets, desc="Runtime filters"):
            samples = index[pdb_id.lower()]
            for sample in range(len(samples)):
                s = samples[sample]
                for ligand in s:
                    # Enforce SMILES filter.
                    skip = False
                    for lig in self.moad[pdb_id].ligands:
                        if lig.name == ligand and lig.smiles not in self.split.smiles:
                            skip = True
                            break

                    if skip:
                        continue

                    frags = s[ligand]
                    for frag_idx in range(len(frags)):
                        if frags[frag_idx] != 0:
                            internal_index.append(
                                MOADFragmentDataset_entry(
                                    pdb_id=pdb_id,
                                    sample=sample,
                                    ligand=ligand,
                                    frag_idx=frag_idx,
                                )
                            )

        self._fragment_index = index
        self._internal_index = internal_index
        self.has_index = True

    def __len__(self) -> int:
        assert self.has_index, "Call MOADFragmentDataset.index() first"
        return len(self._internal_index)

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol, Mol]:
        """Returns (receptor, parent, fragment)"""
        assert self.has_index, "Call MOADFragmentDataset.index() first"
        assert idx >= 0 and idx <= len(self), "Index out of bounds"

        entry = self._internal_index[idx]

        receptor, ligands = self.moad[entry.pdb_id][entry.sample]

        for ligand in ligands:
            if ligand.meta["moad_ligand"].name == entry.ligand:
                pairs = ligand.split_bonds()
                parent, fragment = pairs[entry.frag_idx]
                break
        else:
            raise Exception("Ligand not found")

        sample = (receptor, parent, fragment)

        if self.transform:
            return self.transform(*sample)
        else:
            return sample
