"""Test the accuracy of the model on the test set, per fragment."""

import rdkit
from rdkit import Chem
import json
import sys
import pickle
import os
from tqdm import tqdm

test_json_filename = sys.argv[1]

# Load the test data
print("Loading test data...")
if os.path.exists(f"{test_json_filename}.pkl"):
    with open(f"{test_json_filename}.pkl", "rb") as f:
        data = pickle.load(f)
else:
    with open(test_json_filename, "r") as f:
        data = json.load(f)
    with open(f"{test_json_filename}.pkl", "wb") as f:
        pickle.dump(data, f)

result_counts = {}

for entry in tqdm(data["entries"], desc="Collecting results"):
    correct_smi = entry["groundTruth"]["fragmentSmiles"]
    top_predictions = [
        entry["avgOfCheckpoints"]["closestFromLabelSet"][i]["smiles"] for i in range(4)
    ]

    # Use RDKit to make sure both smiles are cannonical
    correct_smi = Chem.MolToSmiles(Chem.MolFromSmiles(correct_smi))
    top_predictions = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in top_predictions]

    if correct_smi not in result_counts:
        result_counts[correct_smi] = {
            "top-1": {"correct": 0, "incorrect": 0,},
            "top-4": {"correct": 0, "incorrect": 0,},
        }
    if correct_smi == top_predictions[0]:
        result_counts[correct_smi]["top-1"]["correct"] += 1
    else:
        result_counts[correct_smi]["top-1"]["incorrect"] += 1
    if correct_smi in top_predictions:
        result_counts[correct_smi]["top-4"]["correct"] += 1
    else:
        result_counts[correct_smi]["top-4"]["incorrect"] += 1

# Calculate accuracy for each correct smiles
accuracy_per_smiles_lst = []
for correct_smi, counts in result_counts.items():
    total = counts["top-1"]["correct"] + counts["top-1"]["incorrect"]
    accuracy_top_1 = counts["top-1"]["correct"] / total
    accuracy_top_4 = counts["top-4"]["correct"] / total
    accuracy_per_smiles_lst.append((correct_smi, accuracy_top_1, accuracy_top_4, total))

# Sort by accuracy
accuracy_per_smiles_lst = sorted(
    accuracy_per_smiles_lst, key=lambda x: x[1], reverse=True
)

# Print the results
with open(f"{test_json_filename}.accuracy_per_smiles.txt", "w") as f:
    print("SMILES\tAccuracy Top-1\tAccuracy Top-4\tTotal Count")
    for correct_smi, accuracy_top_1, accuracy_top_4, total in accuracy_per_smiles_lst:
        print(f"{correct_smi}\t{accuracy_top_1}\t{accuracy_top_4}\t{total}")
        f.write(f"{correct_smi}\t{accuracy_top_1}\t{accuracy_top_4}\t{total}\n")

print("")
print("Results written to:")
print(f"{test_json_filename}.accuracy_per_smiles.txt")
print("")
