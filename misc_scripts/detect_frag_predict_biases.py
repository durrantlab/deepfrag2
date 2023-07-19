import sys
import json
from collections import defaultdict
import pandas as pd


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    correct_frag_smiles = [
        entry["correct"]["fragmentSmiles"] for entry in data["entries"]
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
            "correct": v,  #  / total,

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

            correct_smiles = entry["correct"]["fragmentSmiles"]
            key = f"accuracy_{n}"
            if correct_smiles in closest_smiles:
                data_catalogued[correct_smiles][key] += 1

    # Convert all the entries to percentages
    total = sum(correct_frag_counts.values())
    for entry in data_catalogued.values():
        entry["correct"] = 100 * entry["correct"] / total
        for n in top_k_ns:
            key = f"present_{n}"
            entry[key] = 100 * entry[key] / total
            key = f"accuracy_{n}"
            entry[key] = 100 * entry[key] / total


    # Convert it to a dataframe
    df = pd.DataFrame.from_dict(data_catalogued, orient="index")

    # Sort by accuracy_4 from largest to smallest value
    df = df.sort_values(by="accuracy_4", ascending=False)

    # Add a column called "present_4_enrichment" that is "present_4" divided by
    # "correct"
    df["present_4_enrichment"] = df["present_4"] / df["correct"]

    # Same for accuracy_4
    df["accuracy_4_enrichment"] = df["accuracy_4"] / df["correct"]

    # Save as a csv string
    data_csv = df.to_csv()

    # Collect other statistics.
    num_uniq_frags = len(df)
    num_frags_never_accurately_predicted_top_4 = len(df[df["accuracy_4"] == 0])
    num_frags_never_present_in_predictions_top_4 = len(df[df["present_4"] == 0])

    data_csv = f"""unique frags:,{num_uniq_frags}
never accurately predicted in top 4:,{num_frags_never_accurately_predicted_top_4}
never present in top-4 predictions regardless of accuracy:,{num_frags_never_present_in_predictions_top_4}

Note: values below are percents.

"correct" is the actual fragment derived from the source structures.
"present_1" is the percent of the time the fragment is the top prediction even if it is not the correct fragment.
"present_4" is the percent of the time the fragment is in the top 4 predictions even if it is not the correct fragment.
"accuracy_1" is the percent of the time the fragment is correctly selected as the top fragment.
"accuracy_4" is the percent of the time the fragment is correctly selected as one of the top 4 fragments.
"present_4_enrichment" is "present_4" divided by "correct".
"accuracy_4_enrichment" is "accuracy_4" divided by "correct".

""" + data_csv

    # import pdb; pdb.set_trace()

    # df.to_csv("frag_smiles_biases.csv")

    with open("frag_smiles_biases.csv", "w") as f:
        f.write(data_csv)



if __name__ == "__main__":
    main()
