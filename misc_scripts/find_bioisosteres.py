# Given the json output file (test_results.json), find candidate bioisosteres.

import json
import sys
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from mpire import WorkerPool
from mpire.utils import make_single_arguments
import numpy
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors


def fix_smi(orig_smi):
    smi = re.sub(r"\[[0-9]{0,5}\*\]", "*", orig_smi, 0, re.MULTILINE)
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol)

    changed = True
    while changed:
        changed = False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "H":
                continue

            if atom.GetFormalCharge() == -1:
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                atom.SetFormalCharge(0)
                changed = True
            elif atom.GetFormalCharge() == 1:
                for bond in atom.GetBonds():
                    bondedAtoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
                    bondedHydrogens = [a for a in bondedAtoms if a.GetSymbol() == "H"]
                    if len(bondedHydrogens) == 0:
                        continue
                    atom.SetFormalCharge(0)
                    bondedHydrogen = bondedHydrogens[0]
                    emol = Chem.EditableMol(mol)
                    try:
                        emol.RemoveAtom(bondedHydrogen.GetIdx())
                    except:
                        print("Couldn't remove atom:", Chem.MolToSmiles(mol))
                    mol = emol.GetMol()
                    changed = True
                    break

    try:
        mol = Chem.RemoveHs(mol)
    except:
        print("Could not remove hydrogens:", Chem.MolToSmiles(mol))
        pass

    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

    # print(orig_smi, "\n", smi, "\n")
    return smi


filename = sys.argv[1]
data = json.load(open(filename))

TANIMOTO_CUTOFF = 0.2
HEAVY_ATOM_COUNT_CUTOFF = 5

def get_score(entry):
    correct_smi = fix_smi(entry["correct"]["fragmentSmiles"])
    closest_smiles = [
        fix_smi(e["smiles"]) for e in entry["averagedPrediction"]["closestFromLabelSet"]
    ]

    results = []
    for i, closest_smi in enumerate(closest_smiles):
        if correct_smi != closest_smi:
            srted = "\t".join(sorted([correct_smi, closest_smi]))
            results.append(
                (srted, 1 - i / len(closest_smiles))
            )

    return results

    # if correct_smi == closest_smiles[0]:
    #     return 1
    #     # print(correct_smi, closest_smiles[0])
    # return 0


with WorkerPool(n_jobs=12) as pool:
    results = pool.map(
        get_score,
        make_single_arguments(data["entries"], generator=False),
        progress_bar=True,
    )

colated = {}
for result in results:
    for r, score in result:
        if r not in colated.keys():
            colated[r] = 0
        colated[r] += score

colated = [(v, *k.split("\t")) for k, v in colated.items()]
colated.sort(key=lambda x: x[0], reverse=True)


def calc_tanimoto(score, smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)

    # Calculate the tanimoto similarity (tanimoto is default:
    # https://www.rdkit.org/docs/GettingStartedInPython.html)
    return DataStructs.FingerprintSimilarity(
        Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2)
    )

with WorkerPool(n_jobs=12) as pool:
    tanimotos = pool.map(calc_tanimoto, colated, progress_bar=True,)

colated = [[*c, t] for c, t in zip(colated, tanimotos) if t < TANIMOTO_CUTOFF]

def mol_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcExactMolWt(mol)

def num_heavy_atoms(score, smi1, smi2, tanimoto) -> int:
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    return max(mol1.GetNumHeavyAtoms(), mol2.GetNumHeavyAtoms())

with WorkerPool(n_jobs=12) as pool:
    heavy_atom_counts = pool.map(num_heavy_atoms, colated, progress_bar=True,)

colated = [[*c, cnt] for c, cnt in zip(colated, heavy_atom_counts) if cnt > HEAVY_ATOM_COUNT_CUTOFF]



# print(numpy.sum(results), "/", len(results))
import pdb

pdb.set_trace()
