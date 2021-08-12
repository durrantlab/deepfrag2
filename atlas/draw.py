
from typing import Tuple

import py3Dmol


class DrawContext(object):
    """A DrawContext is a thin wrapper over a py3Dmol.view that provides some
    helper methods for drawing molecular structures."""
    
    def __init__(self, width=600, height=600, **kwargs):
        self._view = py3Dmol.view(width=width, height=height, **kwargs)

    @property
    def view(self):
        return self._view

    def draw_mol(self, mol: 'MolGraph'):
        self._view.addModel(mol.to_sdf(), 'sdf')
        self._view.setStyle({'stick':{}})
        self._view.zoomTo()

    def draw_sphere(
        self, 
        center: Tuple[float, float, float],
        radius: float = 1,
        color: str = 'green',
        opacity: float = 1
    ):
        self._view.addSphere({
            'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
            'radius': float(radius),
            'color': color,
            'opacity': opacity
        })
        
    def render(self) -> 'py3Dmol.view':
        return self._view.render()
