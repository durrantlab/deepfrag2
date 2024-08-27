import pandas as pd
import matplotlib.pyplot as plt
import os
import json

num_epochs = 60
result_paths = {
    "1.rdk10": "path/to/json/files",
    "2.rdk10Morgan": "path/to/json/files",
    "3.molbertBinary": "path/to/json/files",
}

data_sorted_by_topk = {
    "testTop1": {},
    "testTop8": {},
    "testTop16": {},
    "testTop32": {},
    "testTop64": {},
}

for key in result_paths:
    path = result_paths[key]
    all_files = os.listdir(path)
    json_files = [
        os.path.abspath(os.path.join(path, f)) for f in all_files if f.endswith(".json")
    ]

    for json_file in json_files:
        with open(json_file) as f:
            json_result_inf = json.load(f)
        f.close()

        entries = json_result_inf["checkpoints"][0]
        idx_ckpt = int(entries["name"].split("loss-epoch=")[1].split("-loss")[0])
        topk = entries["topK"]
        for key_top_k in data_sorted_by_topk:
            top_k = data_sorted_by_topk[key_top_k]
            if key not in top_k:
                top_k[key] = [0] * num_epochs
            top_k[key][idx_ckpt] = float(topk[key_top_k])

for key_top_k in data_sorted_by_topk:
    data = data_sorted_by_topk[key_top_k]
    df = pd.DataFrame(data=data)
    df.plot(title=key_top_k, xlabel="Epochs", ylabel="Accuracy")
    plt.savefig(os.getcwd() + os.sep + key_top_k + ".png")
