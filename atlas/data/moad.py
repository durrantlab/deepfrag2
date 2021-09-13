from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

import prody

from .mol import Mol


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

    def num_structures(self) -> int:
        """Returns the number of on-disk structures."""
        return len(self.files)

    def __getitem__(self, idx: int):
        """"""
        m = prody.parsePDB(self.files[idx])

        receptor = Mol.from_prody(m.select("not (nucleic or hetatm) and not water"))
        receptor.meta["name"] = f"Receptor {self.pdb_id.lower()}"
        ligands = []

        for lig in self.ligands:
            if lig.is_valid:
                lig_atoms = m.select(
                    f"chain {lig.chain} and resnum >= {lig.resnum} and resnum < {lig.resnum + lig.reslength}"
                )
                lig_mol = Mol.from_prody(lig_atoms, _fix_moad_smiles(lig.smiles))
                lig_mol.meta["name"] = lig.name

                ligands.append((lig_mol, lig))

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
class MOAD_metadata(object):
    classes: List["MOAD_class"]
    _lookup: Dict["str", "MOAD_target"] = field(default_factory=dict)

    def _init_lookup(self):
        for c in self.classes:
            for f in c.families:
                for t in f.targets:
                    self._lookup[t.pdb_id.lower()] = t

    def __getitem__(self, key: str) -> "MOAD_target":
        assert type(key) is str, f"PDB ID must be a str (got {type(key)})"
        k = key.lower()
        assert k in self._lookup, f'Target "{k}" not found.'
        return self._lookup[k]

    @staticmethod
    def load_from_csv(path) -> "MOAD_metadata":
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

        meta = MOAD_metadata(classes=classes)
        meta._init_lookup()
        return meta

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
                        print(f"Warn: no structures for {k}")
