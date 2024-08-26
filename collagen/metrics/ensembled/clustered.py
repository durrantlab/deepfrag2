from typing import Any
import torch
import numpy as np
from scipy.spatial.distance import cdist
import math

# TODO: NOT USED. Part of early efforts to improve accuracy by filtering out
# outlier predictions. Never got it to work. Might be worth revisiting.


def create_initial_prediction_tensor(
    model_after_first_rotation: Any, num_rotations: int, device: torch.device
) -> torch.Tensor:
    num_entries = model_after_first_rotation.predictions.shape[0]
    fp_size = model_after_first_rotation.predictions.shape[1]
    initial_tnsr = torch.zeros(
        size=(num_rotations, num_entries, fp_size), device=device,
    )

    initial_tnsr[0] = model_after_first_rotation.predictions

    return initial_tnsr


def udpate_prediction_tensor(
    existing_tensor: torch.Tensor, data_to_add: torch.Tensor, idx: int
):
    existing_tensor[idx] = data_to_add


# Below also doesn't seem to improve anything.
def finalize_prediction_tensor(
    final_tensor: torch.Tensor, num_rotations: int, device: torch.device
):
    num_entries = final_tensor.shape[1]
    final_arr = final_tensor.cpu().numpy()

    final_tnsr = torch.zeros(size=final_tensor.shape[1:], device=device,)

    for i in range(num_entries):
        # Get the fingerprints for each of the rotations corresponding to this
        # entry
        fps_per_rot = final_arr[:, i]
        avg_fps = np.array([fps_per_rot.sum(axis=0) / float(fps_per_rot.shape[0])])

        # keep only those closest to mean as way of filtering out outliers.
        num_to_keep = math.ceil(fps_per_rot.shape[0] * 0.5)
        dists = cdist(avg_fps, fps_per_rot)
        fps_per_rot = fps_per_rot[np.argsort(dists)[0]][:num_to_keep]

        # Recalculate mean
        avg_fps = fps_per_rot.sum(axis=0) / float(fps_per_rot.shape[0])
        final_tnsr[i] = torch.Tensor(avg_fps)

        # Cluster
        # outliers_fraction = 0.25
        # clf = EllipticEnvelope(contamination=.1)
        # clf.fit(fps_per_rot)
        # y_pred = clf.decision_function(fps_per_rot).ravel()
        # threshold = stats.scoreatpercentile(y_pred,
        #                                     100 * outliers_fraction)
        # import pdb; pdb.set_trace()

        # fps_per_rot_normalized = StandardScaler().fit_transform(fps_per_rot)
        # db = DBSCAN(eps=50, min_samples=1).fit(fps_per_rot_normalized)
        # labels = db.labels_

        # # Identify the label that is not noise and is the largest
        # labels_cnts = {}
        # for l in labels:
        #     if l == -1: continue  # Noise
        #     if l not in labels_cnts: labels_cnts[l] = 0
        #     labels_cnts[l] += 1

        # if len(labels_cnts.keys()) > 0:
        #     # So not everything is an outlier. Keep only those fps_per_rot that
        #     # belong to largest cluster. (Otherwise, uses all points.)
        #     most_common_label = sorted(
        #         [[c, l] for l, c in labels_cnts.items()],
        #         key = lambda x: x[0],
        #         reverse=True
        #     )[0][1]

        #     # Get the average fingerprint of the ones in that cluster.
        #     idx_to_keep = np.nonzero(labels == most_common_label)[0]
        #     print(idx_to_keep)
        #     fps_per_rot = fps_per_rot[idx_to_keep]

        # avg_fps = fps_per_rot.sum(axis=0) / float(fps_per_rot.shape[0])

        # final_tnsr[i] = torch.Tensor(avg_fps)

        # # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # # print(n_clusters_)

    return final_tnsr


# Using DBScan didn't seem to work
# def finalize_prediction_tensor(final_tensor: torch.Tensor, num_rotations: int, device: torch.device):
#     num_entries = final_tensor.shape[1]
#     final_arr = final_tensor.cpu().numpy()

#     final_tnsr = torch.zeros(
#         size=final_tensor.shape[1:],
#         device=device,
#     )

#     for i in range(num_entries):
#         # Get the fingerprints for each of the rotations corresponding to this
#         # entry
#         fps_per_rot = final_arr[:,i]

#         # Cluster
#         fps_per_rot_normalized = StandardScaler().fit_transform(fps_per_rot)
#         # db = DBSCAN(eps=50, min_samples=2).fit(fps_per_rot_normalized)
#         # db = DBSCAN(eps=25, min_samples=2).fit(fps_per_rot_normalized)
#         # db = DBSCAN(eps=10, min_samples=2).fit(fps_per_rot_normalized)
#         db = DBSCAN(eps=500, min_samples=2).fit(fps_per_rot_normalized)
#         labels = db.labels_

#         # Identify the label that is not noise and is the largest
#         labels_cnts = {}
#         for l in labels:
#             if l == -1: continue  # Noise
#             if l not in labels_cnts: labels_cnts[l] = 0
#             labels_cnts[l] += 1

#         if len(labels_cnts.keys()) > 0:
#             # So not everything is an outlier. Keep only those fps_per_rot that
#             # belong to largest cluster. (Otherwise, uses all points.)
#             most_common_label = sorted(
#                 [[c, l] for l, c in labels_cnts.items()],
#                 key = lambda x: x[0],
#                 reverse=True
#             )[0][1]

#             # Get the average fingerprint of the ones in that cluster.
#             idx_to_keep = np.nonzero(labels == most_common_label)[0]
#             print(idx_to_keep)
#             fps_per_rot = fps_per_rot[idx_to_keep]

#         avg_fps = fps_per_rot.sum(axis=0) / float(fps_per_rot.shape[0])

#         final_tnsr[i] = torch.Tensor(avg_fps)

#         # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#         # print(n_clusters_)

#     return final_tnsr
