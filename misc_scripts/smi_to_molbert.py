import numpy as np
import os
import sys
sys.path.append("../")

from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

PATH_MOLBERT_MODEL = "./molbert_model"
PATH_MOLBERT_CKPT = os.path.join(
    PATH_MOLBERT_MODEL,
    "molbert_100epochs/checkpoints/last.ckpt"
)

MOLBERT_MODEL = MolBertFeaturizer(PATH_MOLBERT_CKPT, embedding_type='average-1-cat-pooled', max_seq_len=200, device='cuda')


mn = 1e100
mx = -1e100

lines = open(sys.argv[1]).readlines()
t = len(lines)
for i, line in enumerate(lines):
    smi = line.strip()
    print(i, "/", t, ":", smi)
    fp = MOLBERT_MODEL.transform_single(smi)
    n_fp = np.array(fp[0][0])

    n = np.min(n_fp)
    x = np.max(n_fp)

    if n < mn:
        mn = n
    if x > mx:
        mx = x

    print("    min:", mn)
    print("    max:", mx)

import pdb; pdb.set_trace()
