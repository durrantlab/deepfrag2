"""MOAD utilities."""


def fix_smiles(smi: str) -> str:
    """Fix some of the MOAD smiles that are not valid.

    Args:
        smi (str): SMILES string.

    Returns:
        str: Fixed SMILES string.
    """
    return (
        smi.replace("+H3", "H3+")
        .replace("+H2", "H2+")
        .replace("+H", "H+")
        .replace("-H", "H-")
        .replace("Al-11H0", "Al-")  # Strange smiles in pdb 2WZC
    )
