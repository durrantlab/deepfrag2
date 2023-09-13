import os
import csv

path = "pth/to/log/files"

logs = {
    "1_fail_match_FirstSmiles_PDBLigand.log": {
        "pos_info_receptor": 6,
        "pos_info_ligand": 11,
        "results": {}
    },
    "2_fail_match_SecondSmiles_PDBLigand.log": {
        "pos_info_receptor": 6,
        "pos_info_ligand": 11,
        "results": {}
    },
    "3_ligand_not_contain_parent.log":  {
        "pos_info_receptor": 5,
        "pos_info_ligand": 10,
        "results": {}
    },
    "4_ligand_not_contain_first-frag.log":  {
        "pos_info_receptor": 5,
        "pos_info_ligand": 10,
        "results": {}
    },
    "5_ligand_not_contain_second-frag.log":  {
        "pos_info_receptor": 5,
        "pos_info_ligand": 10,
        "results": {}
    },
    "6_error_getting_3d_coordinates_for_parent.log":  {
        "pos_info_receptor": 16,
        "pos_info_ligand": 9,
        "results": {}
    },
    "7_error_getting_3d_coordinates_for_first-frag.log":  {
        "pos_info_receptor": 16,
        "pos_info_ligand": 9,
        "results": {}
    },
    "8_error_getting_3d_coordinates_for_second-frag.log":  {
        "pos_info_receptor": 16,
        "pos_info_ligand": 9,
        "results": {}
    },
    "9_finally_used.log": {
        "pos_info_receptor": 6,
        "pos_info_ligand": 10,
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
