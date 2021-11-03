from typing import Tuple

import k3d
import numpy as np
import py3Dmol


class VoxelView(object):
    @staticmethod
    def draw(tensor: "torch.Tensor", color_map: list = None) -> "k3d.plot.Plot":
        points = []
        opacities = []
        colors = []

        c, x, y, z = np.where(tensor[0] > 0.001)
        opacities = tensor[0][c, x, y, z]
        points = np.stack([x, y, z]).transpose()

        if color_map is None:
            colors = [0 for x in c]
        else:
            colors = [color_map[x] for x in c]

        plot = k3d.plot()

        plot += k3d.points(points, colors=colors, point_sizes=opacities)

        plot.grid = [0, tensor.shape[-3], 0, tensor.shape[-2], 0, tensor.shape[-1]]
        plot.grid_auto_fit = False
        plot.display()


class MolView(object):
    """A DrawContext is a thin wrapper over a py3Dmol.view that provides some
    helper methods for drawing molecular structures."""

    def __init__(self, width=600, height=600, **kwargs):
        self._view = py3Dmol.view(width=width, height=height, **kwargs)

    @property
    def view(self):
        return self._view

    def add_cartoon(self, mol: "Mol", style: dict = {}):
        self._view.addModel(mol.pdb(), "pdb")
        self._view.setStyle({"model": -1}, {"cartoon": style})

    def add_stick(self, mol: "Mol", style: dict = {}):
        self._view.addModel(mol.sdf(), "sdf")
        self._view.setStyle({"model": -1}, {"stick": style})

    def add_sphere(self, mol: "Mol", style: dict = {}):
        self._view.addModel(mol.sdf(), "sdf")
        self._view.setStyle({"model": -1}, {"sphere": style})

    def draw_sphere(
        self,
        center: Tuple[float, float, float],
        radius: float = 1,
        color: str = "green",
        opacity: float = 1,
    ):
        self._view.addSphere(
            {
                "center": {
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                },
                "radius": float(radius),
                "color": color,
                "opacity": opacity,
            }
        )

    def render(self) -> "py3Dmol.view":
        self._view.zoomTo()
        return self._view.render()
