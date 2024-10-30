"""Fingerprinting functions for molecules."""

import numpy as np  # type: ignore
import rdkit.Chem.AllChem as Chem  # type: ignore
from rdkit.Chem import DataStructs  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import os
import wget  # type: ignore
import sys
from zipfile import ZipFile
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import rdkit  # type: ignore

PATH_MOLBERT_MODEL = os.path.join(os.getcwd(), "molbert_model")
PATH_MOLBERT_CKPT = os.path.join(
    PATH_MOLBERT_MODEL,
    f"molbert_100epochs{os.sep}checkpoints{os.sep}last.ckpt",
)

MOLBERT_MODEL = None


def bar_progress(current: float, total: float, width=80):
    """Progress bar for downloading Molbert model.

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
    """Download Molbert model checkpoint."""
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


def _Morgan(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Morgan fingerprints.

    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule (not used).
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.

    Returns:
        np.array: Fingerprint.
    """
    array = np.zeros((0,))
    try:
        assert m is not None, "molecule as parameter is None"
        DataStructs.ConvertToNumpyArray(
            AllChem.GetHashedMorganFingerprint(m, 3, nBits=size),
            array,
        )
    except BaseException as e:
        try:
            assert smiles is not None, "smiles as parameter is None"
            DataStructs.ConvertToNumpyArray(
                AllChem.GetHashedMorganFingerprint(
                    Chem.MolFromSmiles(smiles), 3, nBits=size
                ),
                array,
            )
        except BaseException as e:
            print(
                "Error calculating Morgan Fingerprints on "
                + smiles
                + " because of "
                + str(e),
                file=sys.stderr,
            )
            array = np.zeros((size,))

    return array


def _rdk10_x_morgan(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Creates a vector fusing RDK and Morgan Fingerprints.

    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string (not used).

    Returns:
        np.array: Fingerprint.
    """
    rdk10_vals = _rdk10(m, size, smiles)
    morgan_vals = _Morgan(m, size, smiles)
    rdk10_morgan_vals = np.add(rdk10_vals, morgan_vals)
    rdk10_morgan_vals[rdk10_morgan_vals > 0] = 1
    return rdk10_morgan_vals


@lru_cache
def _molbert(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Molbert fingerprints.

    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule (not used).
        size (int): Size of the fingerprint (not used).
        smiles (str): SMILES string.

    Returns:
        np.array: Fingerprint.
    """
    global MOLBERT_MODEL

    # Make sure MOLBERT_MODEL is not None, using assert
    assert MOLBERT_MODEL is not None, "Molbert model is not loaded"

    fp = MOLBERT_MODEL.transform_single(smiles)
    return np.array(fp[0][0])


def _molbert_shuffled(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str):
    molbert_fp = _molbert(m, size, smiles)
    np.random.shuffle(molbert_fp)
    return molbert_fp


def _molbert_binary(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str) -> np.array:
    """Molbert fingerprints with positive values. Any value less than 0 is just
    set to 0.

    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.

    Returns:
        np.array: Fingerprint.
    """
    try:
        molbert_fp = _molbert(m, size, smiles)
        molbert_fp[molbert_fp <= 0] = 0
        molbert_fp[molbert_fp > 0] = 1
        return molbert_fp
    except Exception as e:
        raise Exception("Error calculating binary molbert fingerprints " + str(e))


def _random_binary(m: "rdkit.Chem.rdchem.Mol", size_: int, smiles: str) -> np.array:
    """Random binary fingerprints to validate non-random correlation .

    Args:
        m (rdkit.Chem.rdchem.Mol): RDKit molecule.
        size (int): Size of the fingerprint.
        smiles (str): SMILES string.

    Returns:
        np.array: Random Fingerprint.
    """
    prob0 = np.random.uniform(0.1, 1.0)
    prob1 = 1 - prob0
    random_fp = np.random.choice([0, 1], size=(size_,), p=[prob0, prob1])
    np.random.shuffle(random_fp)
    return random_fp


FINGERPRINTS = {
    "rdk10": _rdk10,
    "rdk10_x_morgan": _rdk10_x_morgan,
    "random_binary_2048": _random_binary,
    "molbert": _molbert,
    "molbert_binary": _molbert_binary,
    "molbert_shuffled": _molbert_shuffled,
    "random_1536": _random_binary,
}


def fingerprint_for(
    mol: "rdkit.Chem.rdchem.Mol", fp_type: str, size: int, smiles: str
) -> "np.ndarray":
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
