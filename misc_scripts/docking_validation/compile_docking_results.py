import pandas as pd
import argparse
import glob
import os

def extract_docking_score(file_path):
    """
    Extracts the best docking affinity score from a smina .pdbqt_out.pdbqt file.
    """
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("1  "):
                score = line.split()[1]  # the score is the 2nd item in the line
                return float(score)
    return None

def extract_rmsd(file_path):
    """
    Extracts the rmsd of the first pose (previously calculated)
    """
    file_path = file_path[:-4] + "_out.pdbqt.match.dat"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as file:
        first_line = file.readline()
    return float(first_line.split()[1])

def process_directory(directory):
    """
    Process the given directory for docking data and returns it as a DataFrame.
    """
    data = []

    for cryst_lig_path in glob.glob(f"{directory}/*/*cryst_lig.smi.pdbqt.log"):
        pdb_id = os.path.basename(os.path.dirname(cryst_lig_path))
        cryst_score = extract_docking_score(cryst_lig_path)
        cryst_rmsd = extract_rmsd(cryst_lig_path)

        for batch_dir in glob.glob(f"{directory}/{pdb_id}/batch*"):
            row = {
                "pdb_id": pdb_id,
                "batch": os.path.basename(batch_dir),
                "crystal_score": cryst_score,
                "crystal_rmsd": cryst_rmsd,
            }

            # decoy scores
            for decoy_path in glob.glob(f"{batch_dir}/decoy*.pdbqt.log"):
                score = extract_docking_score(decoy_path)
                key = os.path.basename(decoy_path).split(".")[0]
                row[key] = score
                row[key + "_rmsd"] = extract_rmsd(decoy_path)

            # predicted scores
            for pred_path in glob.glob(f"{batch_dir}/predicted*.pdbqt.log"):
                score = extract_docking_score(pred_path)
                key = os.path.basename(pred_path).split(".")[0]
                row[key] = score
                row[key + "_rmsd"] = extract_rmsd(decoy_path)

            data.append(row)

    return pd.DataFrame(data)


def main(directory):
    df = process_directory(directory)
    df.to_csv("docking_data.csv", index=False)
    print("Data saved to docking_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract docking information and save to CSV using pandas"
    )
    parser.add_argument("directory", help="Directory containing the docking files")
    args = parser.parse_args()

    main(args.directory)
