import os
import csv

path = "pth/to/log/files"

logs = {
    "0_matched_assay_pdb-lig.log": {
        "pos_info_receptor": 6,
        "pos_info_ligand": 10,
        "results": {}
    },
    "1_unmatched_assay_pdb-lig.log": {
        "pos_info_receptor": 6,
        "pos_info_ligand": 11,
        "results": {}
    },
    "2_unmatched_pdb-lig_fragment.log":  {
        "pos_info_receptor": 5,
        "pos_info_ligand": 10,
        "results": {}
    },
    "3_error_getting_3d_coordinates.log":  {
        "pos_info_receptor": 16,
        "pos_info_ligand": 9,
        "results": {}
    },
}

for log_file_name in logs:
    log_file = os.path.abspath(os.path.join(path, log_file_name))
    with open(log_file) as reader_log:
        for line in reader_log:
            if line.startswith("INFO"):
                split_line = line.split(" ")
                info_receptor = split_line[logs[log_file_name]["pos_info_receptor"]].replace("\n", "").replace("\"", "")
                info_ligand = split_line[logs[log_file_name]["pos_info_ligand"]].replace("\n", "").replace("\"", "")

                structure = logs[log_file_name]["results"]
                if info_receptor not in structure:
                    structure[info_receptor] = {}

                if info_ligand not in structure[info_receptor]:
                    structure[info_receptor][info_ligand] = 1
                else:
                    structure[info_receptor][info_ligand] = structure[info_receptor][info_ligand] + 1

for log_file_name in logs:
    csv_file = os.path.abspath(os.path.join(path, log_file_name + ".csv"))
    with open(csv_file, 'w') as file:
        csvwriter = csv.writer(file)
        rows = []
        structure = logs[log_file_name]["results"]
        for receptor in structure:
            for ligand in structure[receptor]:
                rows.append([receptor, ligand, structure[receptor][ligand]])
        csvwriter.writerows(rows)
