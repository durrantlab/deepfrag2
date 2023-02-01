import numpy as np
import rdkit.Chem.AllChem as Chem
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import os
import wget
import sys
from zipfile import ZipFile

PATH_MOLBERT_MODEL = os.path.join(os.getcwd(), "molbert_model")
PATH_MOLBERT_CKPT  = os.path.join(PATH_MOLBERT_MODEL, "molbert_100epochs" + os.sep + "checkpoints" + os.sep + "last.ckpt")


def bar_progress(current, total, width=80):
    progress_message = "Downloading Molbert model: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_molbert_ckpt():
    if not os.path.exists(PATH_MOLBERT_CKPT):
        os.makedirs(PATH_MOLBERT_MODEL, exist_ok=True)
        file_name = wget.download("https://ndownloader.figshare.com/files/25611290", PATH_MOLBERT_MODEL + os.sep + "model.zip", bar_progress)
        with ZipFile(file_name, 'r') as zObject:
            zObject.extractall(path=os.fspath(PATH_MOLBERT_MODEL))
            zObject.close()
        os.remove(file_name)

    global MOLBERT_MODEL
    MOLBERT_MODEL = MolBertFeaturizer(PATH_MOLBERT_CKPT, embedding_type='average-1-cat-pooled', max_seq_len=200, device='cpu')


def _rdk10(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str):
    """RDKFingerprint with maxPath=10."""

    fp = Chem.rdmolops.RDKFingerprint(m, maxPath=10, fpSize=size)
    n_fp = list(map(int, list(fp.ToBitString())))
    return np.array(n_fp)


def _molbert(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str):
    smiles = smiles.replace('*', '')
    fp = MOLBERT_MODEL.transform_single(smiles)
    n_fp = np.array(fp[0][0])
    return n_fp


FINGERPRINTS = {"rdk10": _rdk10, "molbert": _molbert}


def fingerprint_for(
    mol: "rdkit.Chem.rdchem.Mol", fp_type: str, size: int, smiles: str
) -> "numpy.ndarray":
    """Compute a fingerprint for an rdkit mol. Raises an exception if the
    fingerprint is not found."""

    if fp_type in FINGERPRINTS:
        return FINGERPRINTS[fp_type](mol, size, smiles)

    raise Exception(
        "Fingerprint %s not found. Available: %s"
        % (fp_type, repr([k for k in FINGERPRINTS]))
    )
