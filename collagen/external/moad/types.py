
from dataclasses import dataclass, field
from collagen.core import args as user_args
from typing import List, Tuple
from pathlib import Path
import textwrap
from collagen.core.mol import TemplateGeometryMismatchException, UnparsableGeometryException, UnparsableSMILESException
from rdkit import Chem
import os
import pickle
from io import StringIO 
# from functools import lru_cache

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
    cache_pdbs: bool = False
    grid_width: int = 24  # Number of grid points in each dimension.
    grid_resolution: float = 0.75  # Distance between neighboring grid points in angstroms.
    noh: bool = False  # If true, discards hydrogen atoms
    no_distant_atoms: bool = False
    
    # DeepFrag requires 3.062500, but a few extra angstroms won't hurt. Note
    # that this is effectively hard coded because never specified elsewhere.
    # Should be from max((atom_radii[i] * atom_scale)**2)
    grid_padding: float = 6.0  

    def __len__(self) -> int:
        """Returns the number of on-disk structures."""
        return len(self.files)

    # memory_cache = {}

    def _get_cache_if_available(self, idx):
        # Load the protein/ligand complex (PDB formatted).
        
        # pkl_filename = str(self.files[idx]) + ".pkl"
        pkl_filename = str(self.files[idx])
        if self.no_distant_atoms: 
            pkl_filename += "_" + str(self.grid_width) + "_" + str(self.grid_resolution)
        if self.noh:
            pkl_filename += "_noh"
        pkl_filename += ".pkl"

        if self.cache_pdbs and os.path.exists(pkl_filename):
            # Get if from the pickle.
            try:
                with open(pkl_filename, "rb") as f:
                    payload = pickle.load(f)  # [receptor, ligands]
                    # print("Loaded from pkl")
                    # self.memory_cache[idx] = payload
                    return payload, pkl_filename
            except:
                # If there's an error loading the pickle file, regenerate the
                # pickle file
                # print("Corrupt pkl")
                pass
        return None, pkl_filename

    def _load_pdb(self, idx):
        # If you get here, either you're not caching PDBs, or there is no
        # previous cached PDB you can use, or that cached file is corrupted.
        if self.cache_pdbs:
            # To improve caching, consider only lines that start with ATOM or
            # HETATM, etc. This makes files smaller and speeds pickling a bit.
            with open(self.files[idx]) as f:
                lines = [
                    l for l in f.readlines() 
                    if l.startswith("ATOM") or l.startswith("HETATM") or l.startswith("MODEL") or l.startswith("END")
                ]
            pdb_txt = "".join(lines)

            # Also, keep only the first model.
            pdb_txt = pdb_txt.split("ENDMDL")[0]
            m = prody.parsePDBStream(StringIO(pdb_txt), model=1)
        else:
            # Not caching, so just load the file without preprocessing.
            f = open(self.files[idx], "r")
            m = prody.parsePDBStream(f, model=1)  # model=1 not necessary, but just in case...
            f.close()

        return m

    def _make_lig_mol(self, lig_atoms, lig):
        try:
            lig_mol = Mol.from_prody(
                lig_atoms, fix_moad_smiles(lig.smiles), sanitize=True
            )
            lig_mol.meta["name"] = lig.name
            lig_mol.meta["moad_ligand"] = lig

            return lig_mol
        except UnparsableSMILESException as err:
            if user_args.verbose:
                msg = str(err).replace("[LIGAND]", self.pdb_id + ":" + lig.name)
                print(textwrap.fill(msg, subsequent_indent="  "))
            return None
        except UnparsableGeometryException as err:
            # Some geometries are particularly bad and just can't be parsed. For
            # example, 4P3R:NAP:A:202.
            if user_args.verbose:
                msg = str(err).replace("[LIGAND]", self.pdb_id + ":" + lig.name)
                print(textwrap.fill(msg, subsequent_indent="  "))
            return None
        except TemplateGeometryMismatchException as err:
            # Note that at least some of the time, this is due to bad data from
            # MOAD. See, for example, BindingMoad2019/3i4y.bio1, where a ligand
            # is divided across models for some reason. Some large ligands are
            # also not fully resolved in the crystal structure, so good reason
            # not to include them. For example, 5EIV:HYP GLY PRO HYP GLY PRO
            # HYP:I:-5. # In other cases, there is disagreement about what is a
            # ligand. The PDB entry for 6HMG separates out GLC and GDQ as
            # different ligands, even though they are covalently bonded to one
            # another. Binding MOAD calls this one ligand, "GDQ GLC". I've spot
            # checked a number of these, and the ones that are throw out should
            # be thrown out.
            if user_args.verbose:
                msg = str(err).replace("[LIGAND]", self.pdb_id + ":" + lig.name)
                print(textwrap.fill(msg, subsequent_indent="  "))
            return None
        except Exception as err:
            if user_args.verbose:
                msg = (
                    "WARNING: Could not process ligand " + 
                    self.pdb_id + ":" + lig.name + ". " +
                    "An unknown error occured: " + str(err)
                )
                print(textwrap.fill(msg, subsequent_indent="  "))
            return None

    def _make_rec_mol(self, m, ignore_sels, lig_sels):
        if len(ignore_sels) > 0:
            rec_sel = "not water and not (%s)" % " or ".join(
                f"({x})" for x in ignore_sels
            )
        else:
            rec_sel = "not water"

        if self.no_distant_atoms:
            # Only keep those portions of the receptor that are near some ligand (to
            # speed later calculations).

            all_lig_sel = "(".join(lig_sels) + ")"
            
            # Get half distance along axis
            dist = 0.5 * self.grid_width * self.grid_resolution

            # Need to account for diagnol
            dist = (3 ** 0.5) * dist
            
            # Add padding
            dist = dist + self.grid_padding

            rec_sel = "(" + rec_sel + ") and (exwithin " + str(dist) + " of (" + all_lig_sel + "))"

        if self.noh:
            rec_sel = "not hydrogen and (" + rec_sel + ")"

        rec_mol = Mol.from_prody(m.select(rec_sel))
        rec_mol.meta["name"] = f"Receptor {self.pdb_id.lower()}"

        return rec_mol
    
    def _save_to_cache_if_needed(self, pkl_filename, rec_mol, ligands):
        if self.cache_pdbs:
            with open(pkl_filename, "wb") as f:
                # print("Save pickle")
                pickle.dump([rec_mol, ligands], f, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol]:
        """
        Load the Nth structure for this target.

        Args:
            idx (int): The index of the biological assembly to load.

        Returns a (receptor, ligand) tuple of :class:`atlas.data.mol.Mol` objects.
        """

        # print(len(self.memory_cache.keys()))
        # if idx in self.memory_cache:
        #     print("Getting from cache...")
        #     return self.memory_cache[idx][0], self.memory_cache[idx][1]  # [receptor, ligands]

        cached_recep_and_ligs, pkl_filename = self._get_cache_if_available(idx)
        if cached_recep_and_ligs is not None:
            return cached_recep_and_ligs
        
        m = self._load_pdb(idx)

        ignore_sels = []
        lig_sels = []
        ligands = []

        for lig in self.ligands:
            # Note that "(altloc _ or altloc A)" makes sure only the first
            # alternate locations are used.
            lig_sel = f"chain {lig.chain} and resnum >= {lig.resnum} and resnum < {lig.resnum + lig.reslength} and (altloc _ or altloc A)"

            if lig.validity != "Part of Protein":
                ignore_sels.append(lig_sel)

            if lig.is_valid:
                lig_atoms = m.select(lig_sel)

                # Ligand may not be present in this biological assembly.
                if lig_atoms is None:
                    continue

                lig_mol = self._make_lig_mol(lig_atoms, lig)

                if lig_mol is None:
                    continue

                ligands.append(lig_mol)
                lig_sels.append(lig_sel)

        rec_mol = self._make_rec_mol(m, ignore_sels, lig_sels)

        self._save_to_cache_if_needed(pkl_filename, rec_mol, ligands)

        return rec_mol, ligands


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
