import pandas as pd
import matplotlib.pyplot as plt
import os

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
        with open(json_file, 'r') as file:
            for line in file:
                line = line.strip()

                if "name" in line:
                    idx_ckpt = int(line.split("loss-epoch=")[1].split("-loss")[0])
                    if idx_ckpt >= num_epochs:
                        break
                elif "topK" in line:
                    for line_top_k in file:
                        line_top_k = line_top_k.strip().split(":")
                        if "}" in line_top_k:
                            break

                        key_top_k = str(line_top_k[0].split('\"')[1].split('\"')[0])
                        val_top_k = float(line_top_k[1].split(' ')[1].split(',')[0])
                        top_k = data_sorted_by_topk[key_top_k]
                        if key not in top_k:
                            top_k[key] = [0] * num_epochs
                        top_k[key][idx_ckpt] = val_top_k
                    break
        file.close()

for key_top_k in data_sorted_by_topk:
    data = data_sorted_by_topk[key_top_k]
    df = pd.DataFrame(data=data)
    df.plot(title=key_top_k, xlabel="Epochs", ylabel="Accuracy")
    plt.savefig(os.getcwd() + os.sep + key_top_k + ".png")
