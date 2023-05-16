"""Classes for drawing molecules and voxels."""

from typing import Tuple

import k3d
import numpy as np
import py3Dmol


class VoxelView(object):

    """Useful for visualizing voxels."""

    @staticmethod
    def draw(tensor: "torch.Tensor", color_map: list = None):
        """Draw a 3D tensor using k3d.

        Args:
            tensor (torch.Tensor): The tensor to draw.
            color_map (list, optional): A list of colors to use for each voxel. Defaults to None.
        """
        points = []
        opacities = []
        colors = []

        c, x, y, z = np.where(tensor[0] > 0.001)
        opacities = tensor[0][c, x, y, z]
        points = np.stack([x, y, z]).transpose()

        colors = [0 for x in c] if color_map is None else [color_map[x] for x in c]
        plot = k3d.plot()

        plot += k3d.points(points, colors=colors, point_sizes=opacities)

        plot.grid = [0, tensor.shape[-3], 0, tensor.shape[-2], 0, tensor.shape[-1]]
        plot.grid_auto_fit = False
        plot.display()


class MolView(object):

    """A DrawContext is a thin wrapper over a py3Dmol.view that provides some
    helper methods for drawing molecular structures.
    """

    def __init__(self, width=600, height=600, **kwargs):
        """Initialize a molecule viewer.

        Args:
            width (int, optional): the width of the viewer. Defaults to 600.
            height (int, optional): the height of the viewer. Defaults to 600.
        """
        self._view = py3Dmol.view(width=width, height=height, **kwargs)

    @property
    def view(self) -> py3Dmol.view:
        """Return the underlying py3Dmol.view object."""
        return self._view

    def add_cartoon(self, mol: "Mol", style: dict = None):
        """Add a cartoon representation of a molecule to the viewer.
        
        Args:
            mol (Mol): the molecule to draw.
            style (dict, optional): the style to use for the cartoon. Defaults to None.
        """
        if style is None:
            style = {}
        self._view.addModel(mol.pdb(), "pdb")
        self._view.setStyle({"model": -1}, {"cartoon": style})

    def add_stick(self, mol: "Mol", style: dict = None):
        """Add a stick representation of a molecule to the viewer.

        Args:
            mol (Mol): the molecule to draw.
            style (dict, optional): the style to use for the sticks. Defaults to None.
        """
        if style is None:
            style = {}
        self._view.addModel(mol.sdf(), "sdf")
        self._view.setStyle({"model": -1}, {"stick": style})

    def add_sphere(self, mol: "Mol", style: dict = None):
        """Add a sphere representation of a molecule to the viewer.

        Args:
            mol (Mol): the molecule to draw.
            style (dict, optional): the style to use for the spheres. Defaults to None.
        """
        if style is None:
            style = {}
        self._view.addModel(mol.sdf(), "sdf")
        self._view.setStyle({"model": -1}, {"sphere": style})

    def draw_sphere(
        self,
        center: Tuple[float, float, float],
        radius: float = 1,
        color: str = "green",
        opacity: float = 1,
    ):
        """Draw a sphere at a given center with a given radius.

        Args:
            center (Tuple[float, float, float]): the center of the sphere.
            radius (float, optional): the radius of the sphere. Defaults to 1.
            color (str, optional): the color of the sphere. Defaults to "green".
            opacity (float, optional): the opacity of the sphere. Defaults to 1.
        """
        self._view.addSphere(
            {
                "center": {
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                },
                "radius": radius,
                "color": color,
                "opacity": opacity,
            }
        )

    def render(self) -> "py3Dmol.view":
        """Render the molecule viewer.

        Returns:
            py3Dmol.view: the underlying py3Dmol.view object.
        """
        self._view.zoomTo()
        return self._view.render()
