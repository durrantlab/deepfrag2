from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import os


def extract_features():
    path_to_checkpoint = os.getcwd() + os.sep + "molbert_100epochs" + os.sep + "checkpoints" + os.sep + "last.ckpt"
    f = MolBertFeaturizer(path_to_checkpoint)
    features, masks = f.transform(['C'])
    assert all(masks)
    print(features)
