import os
import json
import numpy as np
import pandas as pd
import sys

json_file_path = "path/to/json/results/files"

all_files = os.listdir(json_file_path)
json_files = [os.path.abspath(os.path.join(json_file_path, f)) for f in all_files if f.endswith(".json")]

all_labels_x_json = {}
json_file_names = []

for json_file in json_files:
    filename = os.path.basename(json_file)
    filename_without_ext = filename[:filename.rindex('.')]
    json_file_names.append(filename_without_ext)

    with open(json_file) as f:
        json_result_inf = json.load(f)
    f.close()

    labels = []
    entries_list = json_result_inf["entries"]
    for entry in entries_list:
        receptor_name = entry["correct"]["receptor"]
        parent_smiles = entry["correct"]["parentSmiles"]
        closest_labels = entry["avgOfCheckpoints"]["closestFromLabelSet"]
        for label in closest_labels:
            value_to_insert = receptor_name + "_" + parent_smiles + "_" + label["smiles"]
            labels.append(value_to_insert)

    all_labels_x_json[filename_without_ext] = labels

matrix = np.zeros((len(json_file_names), len(json_file_names)), dtype=np.int32)
for i in range(len(json_file_names)):
    set_main = set(all_labels_x_json[json_file_names[i]])
    for j in range(i + 1, len(json_file_names)):
        set_second = set(all_labels_x_json[json_file_names[j]])
        intersection = set_main.intersection(set_second)
        diff_set_main = set_main - intersection
        diff_set_second = set_second - intersection
        matrix[i][j] = len(diff_set_main)
        matrix[j][i] = len(diff_set_second)

print(json_file_names)
print(matrix)

dataframe = pd.DataFrame(matrix, columns=json_file_names, index=json_file_names)
html_code = dataframe.to_html()

f = open(json_file_path + os.sep + "output.html", "w")
f.write(html_code)
f.close()

sys.exit()
