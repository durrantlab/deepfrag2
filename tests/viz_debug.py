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
    """
    Save the first item in a batch tensor as DX files for each channel.
    
    Args:
        batch_tensor (torch.Tensor): Batch tensor of shape (batch_size, channels, width, height, depth)
        voxel_params (VoxelParams): Voxelization parameters
        center (np.ndarray): Center coordinates used for voxelization
        output_dir (str, optional): Directory to save DX files. Defaults to "debug_viz".
    """

    voxel_params = VoxelParamsDefault.DeepFrag
    center = entry_info.connection_pt
    pdbid = entry_info.receptor_name.split()[-1]
    ligid = entry_info.ligand_id

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_dir + "/info.json", "w") as f:
        json.dump({
            "pdbid": pdbid,
            "ligid": ligid,
            "center": np.array(center).tolist(),
            "fragmentSmiles": entry_info.fragment_smiles,
            "parentSmiles": entry_info.parent_smiles
        }, f)

    # Extract the first item from the batch
    first_item = batch_tensor[0]

    # Compute grid parameters
    nx = ny = nz = voxel_params.width
    spacing = voxel_params.resolution
    # half_width = (voxel_params.width * spacing) / 2.0
    
    # Suppose nx = ny = nz = voxel_params.width
    spacing  = voxel_params.resolution
    nx       = voxel_params.width
    half_box = (nx * spacing) / 2.0

    origin_x = -half_box
    origin_y = -half_box
    origin_z = -half_box
    
    # Save each channel as a separate DX file
    for channel in range(first_item.shape[0]):
        # print("A", first_item.shape, channel)
        grid_data = first_item[channel].cpu().numpy()
        mx = grid_data.max()
        mn = grid_data.min()
        # grid_data = (grid_data - mn) / (mx - mn)

        # Suppose grid_data.shape == (z_dim, y_dim, x_dim)
        # but you want (x_dim, y_dim, z_dim).
        # grid_data = np.transpose(grid_data, (2, 1, 0))
        # grid_data = np.transpose(grid_data, (1, 2, 0))
        # grid_data = np.transpose(grid_data, (0, 2, 1))

        # print("B")
        
        filename = os.path.join(output_dir, f"channel_{channel}.dx")
        with open(filename, "w") as f:
            # OpenDX header
            f.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
            f.write(f"origin {origin_x} {origin_y} {origin_z}\n")
            f.write(f"delta {spacing} 0.0 0.0\n")
            f.write(f"delta 0.0 {spacing} 0.0\n")
            f.write(f"delta 0.0 0.0 {spacing}\n")
            f.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
            f.write(f"object 3 class array type double rank 0 items {nx*ny*nz} data follows\n")
            
            # Write flattened grid data
            for val in grid_data.flatten():
                f.write(f"{val} ")
            
            print(f"Saved channel {channel} to {filename}", "mx", mx, "mn", mn)