"""
This is a utility script to generate a bunch of latent fragment embeddings from
a ZINC database a run a T-SNE projection. You can use this to precompute a CSV
database in the format that viz/fragment_explorer expects.
"""

import argparse
import time

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm

from ..data.zinc import ZINCMolGraphProviderH5
from ..models.dense_graph_autoencoder import DenseGraphAutoencoder


def generate_samples(zinc_path: str, gcn_model: str, num: int, cpu_only: bool):
    print("[*] Loading data...")
    zinc = ZINCMolGraphProviderH5(zinc_path, make_3D=False)

    print("[*] Loading autoencoder model...")
    device = "cpu" if cpu_only else "cuda"
    t_device = torch.device(device)
    mod = DenseGraphAutoencoder.load(gcn_model, device=device)
    mod.models["encoder"].to(t_device)

    print("[*] Generating samples...")

    z_emb = np.zeros((num, mod.args["z_size"]))

    # (smiles, idx, zinc)
    info = []
    curr = 0

    pbar = tqdm(total=num)

    while True:
        pick = np.random.choice(len(zinc))
        g = zinc[pick].to(t_device)

        with torch.no_grad():
            z = mod.encode(g)

        for j in range(len(z)):
            z_emb[curr] = z[j].cpu().numpy()

            info.append((g.smiles, j, g.meta["zinc_id"]))

            pbar.update(1)
            curr += 1

            if curr >= num:
                break

        if curr >= num:
            break

    return z_emb, info


def standardize(data):
    std = np.copy(data)
    std -= std.mean(axis=0)
    std /= std.std(axis=0)
    return std


def make_tsne(z_emb: np.array, pca_embedding_size: int, jobs: int) -> np.array:
    print("[*] Computing PCA...")
    start = time.time()
    z_pre = PCA(pca_embedding_size).fit_transform(z_emb)
    end = time.time()
    print("[*] PCA: %.3f s" % (end - start))

    z_out = np.zeros((len(z_emb), 3))

    print("[*] Computing t-SNE...")
    start = time.time()
    z_out[:, :2] = TSNE(2, verbose=1, n_jobs=jobs).fit_transform(z_pre)
    end = time.time()
    print("[*] t-SNE: %.3f s" % (end - start))

    # Generate color from normalized argmax.
    data_norm = standardize(z_pre)
    z_out[:, 2] = np.argmax(data_norm, axis=1)

    return z_out


def to_csv(arr):
    return "\n".join(",".join([str(x) for x in row]) for row in arr)


def save_csv(csv_path: str, z_out: np.array, info: list):
    csv = [["x", "y", "z", "smiles", "idx", "zinc_id"]]
    for i in range(len(z_out)):
        x, y, z = z_out[i]
        csv.append([float(x), float(y), float(z), info[i][0], info[i][1], info[i][2]])

    with open(csv_path, "w") as f:
        f.write(to_csv(csv))


def run(args):
    print("[1/3] Generating samples")
    z_emb, info = generate_samples(
        args.zinc_path, args.dense_gcn_model, args.num, args.cpu
    )

    print("[2/3] Computing t-SNE")
    z_out = make_tsne(z_emb, args.pca_embedding_size, args.jobs)

    print("[3/3] Generating csv")
    save_csv(args.csv_out, z_out, info)

    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zinc_path", help="Path to zinc.h5")
    parser.add_argument(
        "dense_gcn_model", help="Path to pretrained DenseGraphAutoencoder model"
    )
    parser.add_argument("csv_out", help="Path to out.csv")
    parser.add_argument(
        "--num", default=50000, type=int, help="Number of fragments to use."
    )
    parser.add_argument(
        "--pca_embedding_size", default=50, help="Size of PCA pre-embedding step."
    )
    parser.add_argument(
        "--cpu", default=False, action="store_true", help="Use the CPU only if set."
    )
    parser.add_argument(
        "--jobs", default=1, type=int, help="Number of jobs for the t-SNE step."
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
