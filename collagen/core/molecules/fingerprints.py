import numpy as np
import rdkit.Chem.AllChem as Chem


def _rdk10(m: "rdkit.Chem.rdchem.Mol", size: int):
    """RDKFingerprint with maxPath=10."""

    fp = Chem.rdmolops.RDKFingerprint(m, maxPath=10, fpSize=size)
    n_fp = list(map(int, list(fp.ToBitString())))
    return np.array(n_fp)


FINGERPRINTS = {"rdk10": _rdk10}


def fingerprint_for(
    mol: "rdkit.Chem.rdchem.Mol", fp_type: str, size: int
) -> "numpy.ndarray":
    """Compute a fingerprint for an rdkit mol. Raises an exception if the
    fingerprint is not found."""

    if fp_type in FINGERPRINTS:
        return FINGERPRINTS[fp_type](mol, size)

    raise Exception(
        "Fingerprint %s not found. Available: %s"
        % (fp_type, repr([k for k in FINGERPRINTS]))
    )
