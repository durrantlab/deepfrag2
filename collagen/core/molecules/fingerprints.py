import numpy as np
import rdkit.Chem.AllChem as Chem
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import os
import wget
import sys
from zipfile import ZipFile
from functools import lru_cache

PATH_MOLBERT_MODEL = os.path.join(os.getcwd(), "molbert_model")
<<<<<<< HEAD
PATH_MOLBERT_CKPT = os.path.join(
    PATH_MOLBERT_MODEL,
    f"molbert_100epochs{os.sep}checkpoints{os.sep}last.ckpt",
)

=======
PATH_MOLBERT_CKPT  = os.path.join(PATH_MOLBERT_MODEL, "molbert_100epochs" + os.sep + "checkpoints" + os.sep + "last.ckpt")
MOLBERT_MODEL = None
>>>>>>> af2573cb6d57afad645f6c702a8d4d0e16034995

def bar_progress(current, total, width=80):
    progress_message = "Downloading Molbert model: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_molbert_ckpt():
    global PATH_MOLBERT_CKPT
    global PATH_MOLBERT_MODEL

    if not os.path.exists(PATH_MOLBERT_CKPT):
        os.makedirs(PATH_MOLBERT_MODEL, exist_ok=True)
        file_name = wget.download("https://ndownloader.figshare.com/files/25611290", PATH_MOLBERT_MODEL + os.sep + "model.zip", bar_progress)
        with ZipFile(file_name, 'r') as zObject:
            zObject.extractall(path=os.fspath(PATH_MOLBERT_MODEL))
            zObject.close()
        os.remove(file_name)

    global MOLBERT_MODEL
    MOLBERT_MODEL = MolBertFeaturizer(PATH_MOLBERT_CKPT, embedding_type='average-1-cat-pooled', max_seq_len=200, device='cuda')


def _rdk10(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str):
    """RDKFingerprint with maxPath=10."""

    fp = Chem.rdmolops.RDKFingerprint(m, maxPath=10, fpSize=size)
    n_fp = list(map(int, list(fp.ToBitString())))
    return np.array(n_fp)


@lru_cache
def _molbert(m: "rdkit.Chem.rdchem.Mol", size: int, smiles: str):
    global MOLBERT_MODEL

    #f = open("/var/tmp/tmp.log", "a")
    #f.write("here4 " + smiles + "\n")
    #f.write("here5 " + str(MOLBERT_MODEL) + "\n")
    #f.write("here6 " + str(MOLBERT_MODEL.transform_single) + "\n")
    #f.close()

    # smiles is reasonable, MOLBERT_MODEL and MOLBERT_MODEL.transform_single
    # are defined. No obvious problem.

    # TODO: fp never generated, but no error...
    fp = MOLBERT_MODEL.transform_single(smiles)

    #f.write("here7 " + str(fp) + "\n")
    #f.close()

    n_fp = np.array(fp[0][0])
    #f.write(str(n_fp) + "\n")
    #f.close()
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
