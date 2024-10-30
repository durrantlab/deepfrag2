from dataclasses import dataclass
from typing import List, Set, Union

from collagen.external.common.parent_targets_ligands import Parent_target


@dataclass
class StructuresClass(object):

    """Contains information about a class of structures."""

    ec_num: str
    families: List["StructuresFamily"]


@dataclass
class StructuresFamily(object):

    """Contains informationa bout a family of structures."""

    rep_pdb_id: str
    targets: List[Union[None, "Parent_target"]]


@dataclass
class StructuresSplit(object):

    """Class to hold information about a split of structures in an external
    database."""

    name: str
    targets: List[str]
    smiles: Union[Set[str], List[str]]


@dataclass
class StructureEntry(object):

    """Class to hold information about a specific structure entry."""

    fragment_smiles: str
    parent_smiles: str

    receptor_name: str
    receptor_sequence: str
    connection_pt: List[float]

    # these attributes are used when performing fine-tuning on paired data
    ligand_id: str
    fragment_idx: int

    def hashable_key(self) -> str:
        """Get a hashable key for this entry.

        Returns:
            str: A hashable key for this entry.
        """
        return (
            self.fragment_smiles
            + "--"
            + self.parent_smiles
            + "--"
            + self.receptor_name
            + "--"
            + str(self.connection_pt[0])
            + ","
            + str(self.connection_pt[1])
            + ","
            + str(self.connection_pt[2])
        )
