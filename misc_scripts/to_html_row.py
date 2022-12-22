"""Write a python program to peform the following steps:

1. Import the rdkit library

2. Given a list of smiles strings, create SVG text for each corresponding molecule.

3. Put those SVG texts into the row of an HTML table.
"""

# NOTE: Considers only the first checkpoint.

# Import the rdkit library
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import re
import random
import json
import sys

test_json_filename = sys.argv[1]

pattern = re.compile("<\?xml.*\?>")

def DrawMol(mol, molSize=(450,150), kekulize=True):
    mc = Chem.MolFromSmiles(mol)
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except Exception:
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
    svg_text = [DrawMol(smi.replace("[MATCH]", "")) for smi in list_of_smi]
    # Create a table row
    table_row = "<tr>{}</tr>"
    # Create a list to store the table cells
    table_cells = []
    # For each SVG text
    for i, svg in enumerate(svg_text):
        # Create a table cell
        table_cell = "<td><center>{}</center></td>"
        smi_to_use = list_of_smi[i]
        if "[MATCH]" in smi_to_use:
            smi_to_use = "<span class=\"red-text\">" + smi_to_use.replace("[MATCH]", "") + "</span>"
        # Add the SVG text to the table cell
        table_cell = table_cell.format(f"{svg}<br>{smi_to_use}")
        # Add the table cell to the list
        table_cells.append(table_cell)
    return table_row.format("".join(table_cells))


data = json.load(open(test_json_filename))
lines = []
for entry in data["entries"]:
    correct = entry["correct"]["fragmentSmiles"]
    # Consider only first checkpoint
    ckpt = entry["perCheckpoint"][0]["averagedPrediction"]
    closest = [e["smiles"] for e in ckpt["closestFromLabelSet"]]

    # Try to wrap the correct one in brackets, though this assumes cannonical
    # smiles, which I don't think is the case.
    if correct in closest:
        closest[closest.index(correct)] = f"[MATCH]{correct}"

    lines.append(f"{correct} " + " ".join(closest))
    
num_closest = len(data["entries"][0]["perCheckpoint"][0]["averagedPrediction"]["closestFromLabelSet"])

#lines = open("correct_and_closest.txt").readlines()
random.shuffle(lines)
num_closest = len(lines[0].split()) - 1
print("<style>")
print("table, th, td {")
print("  border: 1px solid black;")
print("  border-collapse: collapse;")
print("}")
print("th, td {")
print("  padding: 2px;")
print("  text-align: center;")
print("  font-size: 24px;")
print("}")
print(".red-text {")
print("  color: red;")
print("  font-weight: bold;")
print("  text-decoration: underline;")
print("}")
print("</style>")

print("<table>")
print("<thead><tr>")
print("<th>Correct</th>")
for i in range(num_closest):
    print(f"<th>Closest {i + 1}</th>")
print("</tr></thead>")
print("<tbody>")
for l in lines[:100]:
    l = l.split()
    print(toHTMLTable(l))
print("</tbody>")
print("</table>")

# print(toHTMLTable(["CC", "CO"]))
# print(toHTMLTable(["CC", "CF"]))
