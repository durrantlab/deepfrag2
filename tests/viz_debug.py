from collagen.core.voxelization.voxelizer import VoxelParams
from collagen.external.common.types import StructureEntry
import torch
import numpy as np
import os
from collagen.core.voxelization.voxelizer import VoxelParamsDefault
import json

def save_batch_first_item_channels(
    batch_tensor: torch.Tensor, 
    entry_info: StructureEntry,
    output_dir: str = "debug_viz"
):
    voxel_params = VoxelParamsDefault.DeepFrag
    center = entry_info.connection_pt
    pdbid = entry_info.receptor_name.split()[-1]
    ligid = entry_info.ligand_id

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump({
            "pdbid": pdbid,
            "ligid": ligid,
            "center": np.array(center).tolist(),
            "fragmentSmiles": entry_info.fragment_smiles,
            "parentSmiles": entry_info.parent_smiles
        }, f)

    first_item = batch_tensor[0]

    nx = ny = nz = voxel_params.width
    spacing = voxel_params.resolution

    # Since we want the grid centered at (0,0,0), place origin at -half_box
    half_box = (nx * spacing) / 2.0
    origin_x = -half_box
    origin_y = -half_box
    origin_z = -half_box

    for channel in range(first_item.shape[0]):
        grid_data = first_item[channel].cpu().numpy()

        # Direct flattening with Fortran order to match DX requirements
        grid_data_flattened = grid_data.ravel(order='F')

        mx = grid_data.max()
        mn = grid_data.min()

        filename = os.path.join(output_dir, f"channel_{channel}.dx")
        with open(filename, "w") as f:
            f.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
            f.write(f"origin {origin_x} {origin_y} {origin_z}\n")
            f.write(f"delta {spacing} 0.0 0.0\n")
            f.write(f"delta 0.0 {spacing} 0.0\n")
            f.write(f"delta 0.0 0.0 {spacing}\n")
            f.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
            f.write(f"object 3 class array type double rank 0 items {nx*ny*nz} data follows\n")

            for val in grid_data_flattened:
                f.write(f"{val} ")
            print(f"Saved channel {channel} to {filename}", "mx", mx, "mn", mn)
