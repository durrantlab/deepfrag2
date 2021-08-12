
import numpy as np
import torch


def hamilton_product(A, B):
    """Apply the hamilton product to vecs with R.
    
    Args:
    - vecs: shape (*,4)
    - R: shape (*,4)
    """
    prod = torch.matmul(A.view(-1,4,1), B.view(-1,1,4))
    
    w = prod[:, 0, 0] - prod[:, 1, 1] - prod[:, 2, 2] - prod[:, 3, 3]
    x = prod[:, 0, 1] + prod[:, 1, 0] - prod[:, 2, 3] + prod[:, 3, 2]
    y = prod[:, 0, 2] + prod[:, 1, 3] + prod[:, 2, 0] - prod[:, 3, 1]
    z = prod[:, 0, 3] - prod[:, 1, 2] + prod[:, 2, 1] + prod[:, 3, 0]
    
    return torch.stack([w, x, y, z], 1)


def apply_quaternion(points, q):
    """Apply quaternion rotation q to each point in points."""
    # Add zero column.
    full_points = torch.cat([
        torch.zeros(len(points),1),
        points.float(), 
    ], 1)
        
    R = q
    Rp = torch.tensor([q[0], -q[1], -q[2], -q[3]])
    
    p1 = hamilton_product(R, full_points)
    p2 = hamilton_product(p1, Rp)
    
    return p2[:,1:]


def random_unit_vector():
    """Returns a random (x,y,z) unit vector."""
    vec = np.random.normal(size=3)
    vec /= np.linalg.norm(vec)
    return vec
