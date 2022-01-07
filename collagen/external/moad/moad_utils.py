
def fix_moad_smiles(smi: str) -> str:
    return (
        smi.replace("+H3", "H3+")
        .replace("+H2", "H2+")
        .replace("+H", "H+")
        .replace("-H", "H-")
        .replace("Al-11H0", "Al-")  # Strange smiles in pdb 2WZC
    )
