"""Write a python program to peform the following steps:

1. Import the rdkit library

2. Given a list of smiles strings, create SVG text for each corresponding molecule.

3. Put those SVG texts into the row of an HTML table.
"""

# Import the rdkit library
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import re
import random

pattern = re.compile("<\?xml.*\?>")

def DrawMol(mol, molSize=(450,150), kekulize=True):
    mc = Chem.MolFromSmiles(mol)
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        Chem.rdDepictor.Compute2DCoords(mc)

    drawer = rdMolDraw2D.MolDraw2DSVG(*molSize)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    svg = re.sub(pattern, '', svg)
    return svg

def toHTMLTable(list_of_smi):
    """
    Put those SVG texts into the row of an HTML table.
    """
    # Create a list to store the SVG text
    svg_text = []
    # For each smiles string in the list
    for smi in list_of_smi:
        # Create a molecule from the smiles string
        # mol = Chem.MolFromSmiles(smi)
        # Create a SVG image of the molecule
        svg_text.append(DrawMol(smi))
    # Create a table row
    table_row = "<tr>{}</tr>"
    # Create a list to store the table cells
    table_cells = []
    # For each SVG text
    for i, svg in enumerate(svg_text):
        # Create a table cell
        table_cell = "<td><center>{}</center></td>"
        # Add the SVG text to the table cell
        table_cell = table_cell.format(svg + "<br>" + list_of_smi[i])
        # Add the table cell to the list
        table_cells.append(table_cell)
    # Join the table cells
    table_row = table_row.format("".join(table_cells))
    # Return the table row
    return table_row


lines = open("correct_and_closest.txt").readlines()
random.shuffle(lines)
print("<table>")
for l in lines[:100]:
    l = l.split()
    print(toHTMLTable(l))
print("</table>")

# print(toHTMLTable(["CC", "CO"]))
# print(toHTMLTable(["CC", "CF"]))
