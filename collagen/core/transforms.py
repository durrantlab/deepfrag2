# import torch
# from collagen.core import Mol, VoxelParams

# Note that this class is only used by the autoencoder app, currently in
# development. Probably not a function you need to look at for now. TODO: not
# used.

# TODO: Latest: Doesn't seem to be used by latest autoencoder either. Let's comment
# out for now.

# class Voxelize(object):

#     """Voxelize a molecule."""

#     def __init__(
#         self, params: VoxelParams, cpu: bool = False, keep_batch: bool = False
#     ):
#         self.params = params
#         self.cpu = cpu
#         self.keep_batch = keep_batch

#     def __call__(self, mol: Mol):
#         if mol.has_coords:
#             t = mol.voxelize(self.params, cpu=self.cpu)
#         else:
#             # RDKit error
#             t = torch.zeros(
#                 size=self.params.tensor_size(), device="cpu" if self.cpu else "cuda"
#             )

#         if not self.keep_batch:
#             t = t[0]
#         return t
