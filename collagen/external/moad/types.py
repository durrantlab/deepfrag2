
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

import prody

from ... import Mol
from .moad_utils import fix_moad_smiles


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

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol]:
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
