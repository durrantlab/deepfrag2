"""Write a python program to peform the following steps:

1. Import the rdkit library

2. Load a list of smiles from a file named smiles.smi

3. Use rdkit to calculate the molecular mass of each smiles.

4. Print the smiles and its mass to the screen."""

# 1. Import the rdkit library
from rdkit import Chem
from rdkit.Chem import Descriptors

# 2. Load a list of smiles from a file named smiles.smi
smiles = open("smiles.smi", "r")

# 3. Use rdkit to calculate the molecular mass of each smiles.
for line in smiles:
    line = line.strip()
    mol = Chem.MolFromSmiles(line)
    mol_mass = Descriptors.ExactMolWt(mol)
    print(line, mol_mass)

# 4. Print the smiles and its mass to the screen.
