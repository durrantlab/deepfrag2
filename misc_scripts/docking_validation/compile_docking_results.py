import pandas as pd
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


# from collections import OrderedDict

def extract_docking_score(file_path):
    """
    Extracts the best docking affinity score from a smina .pdbqt_out.pdbqt file.
    """
    # with open(file_path, "r") as file:
    #     for line in file:
    #         if line.startswith("1  "):
    #             score = line.split()[1]  # the score is the 2nd item in the line
    #             return float(score)
    # return None

    try:
        with open(file_path[:-4] + "_out.pdbqt", "r") as file:
            _ = file.readline()
            second_line = file.readline()
        return float(second_line.split()[2])
    except Exception:
        return None

def extract_rmsd(file_path):
    """
    Extracts the rmsd of the first pose (previously calculated)
    """
    file_path = file_path[:-4] + "_out.pdbqt.match.dat"
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as file:
        first_line = ""
        while not "Score:" in first_line:
            first_line = file.readline()
            if not first_line:
                break

    try:
        return float(first_line.split()[1])
    except Exception:
        return None

def process_directory(directory):
    """
    Process the given directory for docking data and returns it as a DataFrame.
    """
    data = []

    RMSD_CUTOFF = 2.0

    for cryst_lig_path in glob.glob(f"{directory}/*/*cryst_lig.smi.pdbqt.log"):
        pdb_id = os.path.basename(os.path.dirname(cryst_lig_path))
        cryst_score = extract_docking_score(cryst_lig_path)
        cryst_rmsd = extract_rmsd(cryst_lig_path)

        if cryst_rmsd is None or cryst_rmsd > RMSD_CUTOFF:
            # If you can't recapture the crystallographic pose, no use
            # proceeding with decoys/predicteds.
            continue

        for batch_dir in glob.glob(f"{directory}/{pdb_id}/batch*"):
            row = {
                "pdb_id": pdb_id,
                "batch": os.path.basename(batch_dir),
                "crystal_score": cryst_score,
                "crystal_rmsd": cryst_rmsd,
            }

            # decoy scores
            for path in glob.glob(f"{batch_dir}/decoy*.pdbqt.log") + glob.glob(f"{batch_dir}/predicted*.pdbqt.log"):
                rmsd = extract_rmsd(path)
                rmsd_passes_filter = rmsd is not None and rmsd <= RMSD_CUTOFF
                # if rmsd is None or rmsd > RMSD_CUTOFF:
                #     # Don't consider those with different poses from crystal
                #     # structure
                #     continue
                score = extract_docking_score(path)
                key = os.path.basename(path).split(".")[0]
                row["score_" + key] = score
                row["rmsd_" + key] = rmsd
                row["score_if_pass_rmsd_filt_" + key] = score if rmsd_passes_filter else None
                row["delta_score_" + key] = cryst_score - score if score is not None else None
                row["delta_score_if_pass_rmsd_filt_" + key] = cryst_score - score if rmsd_passes_filter and score is not None else None

            # # predicted scores
            # for pred_path in glob.glob(f"{batch_dir}/predicted*.pdbqt.log"):
            #     rmsd = extract_rmsd(pred_path)
            #     if rmsd is None or rmsd > RMSD_CUTOFF:
            #         # Don't consider predicted poses with different poses from
            #         # crystal structure
            #         continue
            #     score = extract_docking_score(pred_path)
            #     key = os.path.basename(pred_path).split(".")[0]
            #     row[key] = score
            #     row[key + "_rmsd"] = rmsd

            # The row should be an ordered dictionary, with the columns in alphabetical order.
            # row = OrderedDict(sorted(row.items()))

            data.append(row)

    return pd.DataFrame(data)

def sort_df_cols(df):
        
    order = [
        "pdb_id",
        "batch",
        "crystal_score",
        "crystal_rmsd",
        "delta_score_predicted1",
        "delta_score_predicted2",
        "delta_score_predicted3",
        "delta_score_predicted4",
        "delta_score_predicted5",
        "delta_score_if_pass_rmsd_filt_predicted1",
        "delta_score_if_pass_rmsd_filt_predicted2",
        "delta_score_if_pass_rmsd_filt_predicted3",
        "delta_score_if_pass_rmsd_filt_predicted4",
        "delta_score_if_pass_rmsd_filt_predicted5",
        "rmsd_predicted1",
        "rmsd_predicted2",
        "rmsd_predicted3",
        "rmsd_predicted4",
        "rmsd_predicted5",
        "score_predicted1",
        "score_predicted2",
        "score_predicted3",
        "score_predicted4",
        "score_predicted5",
        "score_if_pass_rmsd_filt_predicted1",
        "score_if_pass_rmsd_filt_predicted2",
        "score_if_pass_rmsd_filt_predicted3",
        "score_if_pass_rmsd_filt_predicted4",
        "score_if_pass_rmsd_filt_predicted5",
        "delta_score_decoy1",
        "delta_score_decoy2",
        "delta_score_decoy3",
        "delta_score_decoy4",
        "delta_score_decoy5",
        "delta_score_if_pass_rmsd_filt_decoy1",
        "delta_score_if_pass_rmsd_filt_decoy2",
        "delta_score_if_pass_rmsd_filt_decoy3",
        "delta_score_if_pass_rmsd_filt_decoy4",
        "delta_score_if_pass_rmsd_filt_decoy5",
        "rmsd_decoy1",
        "rmsd_decoy2",
        "rmsd_decoy3",
        "rmsd_decoy4",
        "rmsd_decoy5",
        "score_decoy1",
        "score_decoy2",
        "score_decoy3",
        "score_decoy4",
        "score_decoy5",
        "score_if_pass_rmsd_filt_decoy1",
        "score_if_pass_rmsd_filt_decoy2",
        "score_if_pass_rmsd_filt_decoy3",
        "score_if_pass_rmsd_filt_decoy4",
        "score_if_pass_rmsd_filt_decoy5"
    ]

    df = df[order]

    return df

def create_histogram(df, cols1, cols2, label1, label2, bin_edges, file_name="histogram.svg"):
    """
    Create superimposed normalized line histograms from values of two sets of columns in a DataFrame.
    
    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - cols1 (list): First list of columns to extract values from.
    - cols2 (list): Second list of columns to extract values from.
    - label1 (str): Label for the first set of columns.
    - label2 (str): Label for the second set of columns.
    - bin_edges (list): Bin edges for the histogram.
    - file_name (str): Name of the SVG file to save the histogram. Default is 'histogram.svg'.

    Returns:
    None
    """
    
    # Extract values for the first set of columns
    values1 = []
    for col in cols1:
        values1.extend(df[col].dropna().tolist())
    
    # Extract values for the second set of columns
    values2 = []
    for col in cols2:
        values2.extend(df[col].dropna().tolist())

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Compute and plot histogram for cols1
    hist1, edges1 = np.histogram(values1, bins=bin_edges, density=True)
    plt.plot(edges1[:-1], hist1, marker='o', linestyle='-', color='blue', label=label1)
    
    # Compute and plot histogram for cols2
    hist2, edges2 = np.histogram(values2, bins=bin_edges, density=True)
    plt.plot(edges2[:-1], hist2, marker='x', linestyle='-', color='red', label=label2)
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Normalized Histograms')
    plt.legend()
    
    # Save the plot as SVG file
    plt.savefig(file_name, format='svg')
    plt.close()

# Example usage:
# create_histogram(df, ['header1', 'header2'], ['header3', 'header4'], [0, 10, 20, 30, 40, 50])


def main(directory):
    df = process_directory(directory)

    # Order the columns alphabetically
    df = df.reindex(sorted(df.columns), axis=1)
    df = sort_df_cols(df)

    df.to_csv("docking_data.csv", index=False)

    # Make histograms too
    create_histogram(
        df, 
        ['delta_score_predicted1', 'delta_score_predicted2', 'delta_score_predicted3', 'delta_score_predicted4', 'delta_score_predicted5'], 
        ['delta_score_decoy1', 'delta_score_decoy2', 'delta_score_decoy3', 'delta_score_decoy4', 'delta_score_decoy5'], 
        "Predicteds, Delta Score from Cryst",
        "Decoys, Delta Score from Cryst",
        [-20 + i for i in range(31)], # -20 to 10, by 1
        file_name="histogram_delta_score.svg"
    )

    # Also RMSDs
    create_histogram(
        df, 
        ['rmsd_predicted1', 'rmsd_predicted2', 'rmsd_predicted3', 'rmsd_predicted4', 'rmsd_predicted5'], 
        ['rmsd_decoy1', 'rmsd_decoy2', 'rmsd_decoy3', 'rmsd_decoy4', 'rmsd_decoy5'], 
        "Predicteds, RMSD from Cryst",
        "Decoys, RMSD from Cryst",
        # 0 to 15, by 0.5
        [0 + 0.5*i for i in range(31)],
        file_name="histogram_rmsd.svg"
    )

    print("Data saved to docking_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract docking information and save to CSV using pandas"
    )
    parser.add_argument("directory", help="Directory containing the docking files")
    args = parser.parse_args()

    main(args.directory)
