"""Fingerprinting functions for molecules."""

import numpy as np
import rdkit.Chem.AllChem as Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import os
import wget
import sys
from zipfile import ZipFile
from functools import lru_cache

PATH_MOLBERT_MODEL = os.path.join(os.getcwd(), "molbert_model")
PATH_MOLBERT_CKPT = os.path.join(
    PATH_MOLBERT_MODEL, f"molbert_100epochs{os.sep}checkpoints{os.sep}last.ckpt",
)

MOLBERT_MODEL = None

RDKit_DESC_CALC = MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])


def bar_progress(current: float, total: float, width=80):
    """Progress bar for downloading Molbert model. TODO: Candidate for removal.

    Args:
        current (float): Current progress.
        total (float): Total progress.
        width (int, optional): Width of the progress bar. Defaults to 80.
    """
    progress_message = "Downloading Molbert model: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_molbert_ckpt():
    """Download Molbert model checkpoint. TODO: Candidate for removal."""
    global PATH_MOLBERT_CKPT
    global PATH_MOLBERT_MODEL

    if not os.path.exists(PATH_MOLBERT_CKPT):
        os.makedirs(PATH_MOLBERT_MODEL, exist_ok=True)
        file_name = wget.download(
            "https://ndownloader.figshare.com/files/25611290",
            PATH_MOLBERT_MODEL + os.sep + "model.zip",
            bar_progress,
        )
        with ZipFile(file_name, "r") as zObject:
            zObject.extractall(path=os.fspath(PATH_MOLBERT_MODEL))
            zObject.close()
        os.remove(file_name)

    global MOLBERT_MODEL
    MOLBERT_MODEL = MolBertFeaturizer(
        PATH_MOLBERT_CKPT,
        embedding_type="average-1-cat-pooled",
        max_seq_len=200,
        device="cuda",
    )


def _rdk10(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """RDKFingerprint with maxPath=10.

    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string (not used).

    Returns:
        np.array: Fingerprint.
    """
    fp = Chem.rdmolops.RDKFingerprint(m, maxPath=10, fpSize=size)
    n_fp = list(map(int, list(fp.ToBitString())))
    return np.array(n_fp)


def _rdkit_2D_descriptors(
    m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str
) -> np.array:
    """Compute all RDKit Descriptors. TODO: Candidate for removal.

    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule (not used).
        size (int): Size of the fingerprint (not used).
        smiles (str): SMILES string.

    Returns:
        np.array: Fingerprint.
    """
    global RDKit_DESC_CALC
    fp = RDKit_DESC_CALC.CalcDescriptors(mol=Chem.MolFromSmiles(smiles))
    fp = np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0)
    return fp


def _MACCSkeys(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """MACCSkeys fingerprints. TODO: Candidate for removal.
    
    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule (not used).
        size (int): Size of the fingerprint (not used).
        smiles (str): SMILES string.
        
    Returns:
        np.array: Fingerprint.
    """
    fp = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))
    n_fp = list(map(int, list(fp.ToBitString())))
    return np.array(n_fp)


def _Morgan(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Morgan fingerprints. TODO: Candidate for removal.
    
    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule (not used).
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.
        
    Returns:
        np.array: Fingerprint.
    """
    array = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(
        AllChem.GetHashedMorganFingerprint(Chem.MolFromSmiles(smiles), 3, nBits=size),
        array,
    )
    return array


@lru_cache
def _molbert(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Molbert fingerprints. TODO: Candidate for removal.

    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule (not used).
        size (int): Size of the fingerprint (not used).
        smiles (str): SMILES string.

    Returns:
        np.array: Fingerprint.
    """
    global MOLBERT_MODEL
    fp = MOLBERT_MODEL.transform_single(smiles)
    return np.array(fp[0][0])


def _molbert_pos(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Molbert fingerprints with positive values. Any value less than 0 is just
    set to 0. TODO: Candidate for removal.

    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.

    Returns:
        np.array: Fingerprint.
    """
    molbert_fp = _molbert(m, size, smiles)
    molbert_fp[molbert_fp < 0] = 0
    return molbert_fp


def _molbert_norm(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Molbert fingerprints normalized between 0 and 1. TODO: Candidate for removal.
    
    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.
        
    Returns:
        np.array: Fingerprint.
    """
    molbert_fp = _molbert(m, size, smiles)
    mx = np.max(molbert_fp)
    mn = np.min(molbert_fp)
    molbert_fp_norm = np.array([(x - mn) / (mx - mn) for x in molbert_fp])
    molbert_fp_norm = np.nan_to_num(molbert_fp_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return molbert_fp_norm

def _molbert_norm2(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Molbert fingerprints normalized between -6 and 6. Rationale: I calculated
    molbert fingerprints for 7574 unique fragments, and the min/max values for
    any value were -5.631347 and 5.4433527. Let's assume the fingerprint is
    bounded by -6/6. TODO: Candidate for removal.
    
    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.
        
    Returns:
        np.array: Fingerprint.
    """
    molbert_fp = _molbert(m, size, smiles)

    # NOTE: I calculated molbert fingerprints for 7574 unique fragments, and the
    # min/max values for any value were -5.631347 and 5.4433527. Let's assume
    # the fingerprint is bounded by -6/6.

    mx = 6  # np.max(molbert_fp)
    mn = -6  # np.min(molbert_fp)
    molbert_fp_norm2 = np.array([(x - mn) / (mx - mn) for x in molbert_fp])
    molbert_fp_norm2 = np.nan_to_num(molbert_fp_norm2, nan=0.0, posinf=0.0, neginf=0.0)

    return molbert_fp_norm2


def _molbert_x_rdk10(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Molbert fingerprints multiplied by RDKit fingerprints. TODO: Candidate
    for removal.
    
    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.
        
    Returns:
        np.array: Fingerprint.
    """
    rdk10_fp = _rdk10(m, size, smiles)
    molbert_fp = _molbert_norm(m, size, smiles)
    return np.multiply(molbert_fp, rdk10_fp)


def _molbert_x_morgan(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Molbert fingerprints multiplied by Morgan fingerprints. TODO: Candidate
    for removal.
    
    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.

    Returns:
        np.array: Fingerprint.
    """
    morgan_fp = _Morgan(m, size, smiles)
    molbert_fp = _molbert_norm(m, size, smiles)
    return np.multiply(molbert_fp, morgan_fp)


FINGERPRINTS = {
    "rdk10": _rdk10,
    "rdkit_desc": _rdkit_2D_descriptors,
    "maccs": _MACCSkeys,
    "morgan": _Morgan,
    "molbert": _molbert,
    "molbert_pos": _molbert_pos,
    "molbert_norm": _molbert_norm,
    "molbert_sig": _molbert,  # the application of the Sigmoid function is applied in model.py to avoid device conflicts
    "molbert_sig_v2": _molbert,  # the application of the Sigmoid function is applied in model.py to avoid device conflicts
    "molbert_norm2": _molbert_norm2,
    "molbert_x_rdk10": _molbert_x_rdk10,
    "molbert_x_morgan": _molbert_x_morgan,
}


def fingerprint_for(
    mol: "rdkit.Chem.rdchem.Mol", fp_type: str, size: int, smiles: str
) -> "numpy.ndarray":
    """Compute a fingerprint for an rdkit mol. Raises an exception if the
    fingerprint is not found.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule.
        fp_type (str): Fingerprint type.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.
        
    Returns:
        np.array: Fingerprint.
    """
    if fp_type in FINGERPRINTS:
        return FINGERPRINTS[fp_type](mol, size, smiles)

    raise Exception(
        f"Fingerprint {fp_type} not found. Available: {repr(list(FINGERPRINTS))}"
    )
