import sys
import json
from collections import defaultdict
import pandas as pd
import os

# import Pool
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import Draw

precision = 5

def _smiles_to_filename(smiles: str) -> str:
    """Convert a SMILES string into a filename.

    Args:
        smiles (str): The SMILES string.

    Returns:
        str: The filename.
    """
    smiles_hash = "".join([c for c in smiles if c.isalnum()])

    return f"./output/imgs/{smiles_hash}.svg"


def _save_svg(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    filename = _smiles_to_filename(smiles)

    if os.path.exists(filename):
        return

    # Use the correct function to draw the molecule to an SVG
    drawer = Draw.MolDraw2DSVG(100, 100)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Write SVG string to a file
    with open(filename, "w") as f:
        f.write(svg)


def _csv_to_markdown(csv: str, svg_to_first=False) -> str:
    data_markdown = csv.replace(",", "|")
    data_markdown = data_markdown.replace("_", "\_")
    data_markdown = data_markdown.replace("[", "\[")
    data_markdown = data_markdown.replace("]", "\]")
    data_markdown = data_markdown.replace("(", "\(")
    data_markdown = data_markdown.replace(")", "\)")
    data_markdown = data_markdown.replace("\n", "|\n|")
    data_markdown = "|" + data_markdown + "|"

    data_markdown_lines = data_markdown.split("\n")

    num_headers = data_markdown_lines[0].count("|") - 1
    header_line = "|"
    for _ in range(num_headers):
        header_line += " --- |"
    data_markdown_lines.insert(1, header_line)

    if svg_to_first:
        for i, line in enumerate(data_markdown_lines):
            if i <= 1:
                continue
            if i > 500:
                break
            line = line.split("|")
            # import pdb; pdb.set_trace()
            filename = _smiles_to_filename(line[1]).replace("./output/", "./")
            line[1] = f"![{line[1]}]({filename})<br />{line[1]}"
            data_markdown_lines[i] = "|".join(line)

    data_markdown = "\n".join(data_markdown_lines)

    return data_markdown


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    correct_frag_smiles = [
        entry["groundTruth"]["fragmentSmiles"] for entry in data["entries"]
    ]

    top_k_ns = [1, 4]

    # Count the number of times each fragment appears
    correct_frag_counts = defaultdict(int)
    for frag in correct_frag_smiles:
        correct_frag_counts[frag] += 1

    data_catalogued = {
        k: {
            # Keep track of how often it is the correct fragment (reflects
            # database)
            "groundTruth": v,  #  / total,
            # Keep track of how often it is present in the top n, even if not
            # correct (reflects prediciton, can predict in that region of
            # fingerprint space).
            "present_1": 0,
            "present_4": 0,
            # Keep track of how often the correct answer is in the top n.
            "accuracy_1": 0,
            "accuracy_4": 0,
        }
        for k, v in correct_frag_counts.items()
    }

    for n in top_k_ns:
        for entry in data["entries"]:
            # NOTE: Assuming only one checkpoint
            closest_smiles = {
                i["smiles"]
                for i in entry["perCheckpoint"][0]["averagedPrediction"][
                    "closestFromLabelSet"
                ][:n]
            }

            # First, keep track of whether the g
            key = f"present_{n}"
            for smiles in closest_smiles:
                if smiles in data_catalogued:
                    data_catalogued[smiles][key] += 1

            correct_smiles = entry["groundTruth"]["fragmentSmiles"]
            key = f"accuracy_{n}"
            if correct_smiles in closest_smiles:
                data_catalogued[correct_smiles][key] += 1

    # Convert all the entries to percentages
    total = sum(correct_frag_counts.values())
    for entry in data_catalogued.values():
        entry["groundTruth"] = round(100 * entry["groundTruth"] / total, precision)
        for n in top_k_ns:
            key = f"present_{n}"
            entry[key] = round(100 * entry[key] / total, precision)
            key = f"accuracy_{n}"
            entry[key] = round(100 * entry[key] / total, precision)

    # Convert it to a dataframe
    df = pd.DataFrame.from_dict(data_catalogued, orient="index")

    # Sort by accuracy_4 from largest to smallest value
    df = df.sort_values(by="accuracy_4", ascending=False)

    # Add a column called "present_4_enrichment" that is "present_4" divided by
    # "groundTruth"
    df["present_4_enrichment"] = (df["present_4"] / df["groundTruth"]).round(precision)

    # Same for accuracy_4
    df["accuracy_4_enrichment"] = (df["accuracy_4"] / df["groundTruth"]).round(precision)

    # Save as a csv string
    data_csv = df.to_csv()

    # Convert that into a markdown table
    data_markdown = _csv_to_markdown(data_csv, True)

    # Collect other statistics.
    num_uniq_frags = len(df)
    num_frags_never_accurately_predicted_top_4 = len(df[df["accuracy_4"] == 0])
    num_frags_never_present_in_predictions_top_4 = len(df[df["present_4"] == 0])

    summary_csv = f"""Label,Value
unique frags:,{num_uniq_frags}
never accurately predicted in top 4:,{num_frags_never_accurately_predicted_top_4}
never present in top-4 predictions regardless of accuracy:,{num_frags_never_present_in_predictions_top_4}"""

    summary_markdown = _csv_to_markdown(summary_csv)

    notes = """Note: values below are percents.

* "groundTruth" is the actual fragment derived from the source structures.
* "present_1" is the percent of the time the fragment is the top prediction even if it is not the correct fragment.
* "present_4" is the percent of the time the fragment is in the top 4 predictions even if it is not the correct fragment.
* "accuracy_1" is the percent of the time the fragment is correctly selected as the top fragment.
* "accuracy_4" is the percent of the time the fragment is correctly selected as one of the top 4 fragments.
* "present_4_enrichment" is "present_4" divided by "groundTruth".
* "accuracy_4_enrichment" is "accuracy_4" divided by "groundTruth"."""

    csv = f"""{summary_csv}

{notes}

{data_csv}"""

    markdown = f"""{summary_markdown}

{notes}

{data_markdown}"""

    # import pdb; pdb.set_trace()

    # df.to_csv("frag_smiles_biases.csv")

    wrk_dir = "./output/"
    img_dir = f"{wrk_dir}/imgs/"

    # If already exists, delete it.
    if os.path.exists(wrk_dir):
        os.system(f"rm -rf {wrk_dir}")
    os.mkdir(wrk_dir)
    os.mkdir(img_dir)

    # Go through correct smiles and save the images
    for smiles in correct_frag_smiles:
        _save_svg(smiles)

    # Same as above, but using multiple processors
    with Pool(8) as p:
        p.map(_save_svg, correct_frag_smiles)

    with open(f"{wrk_dir}frag_smiles_biases.csv", "w") as f:
        f.write(csv)

    with open(f"{wrk_dir}frag_smiles_biases.md", "w") as f:
        f.write(markdown)


if __name__ == "__main__":
    main()
