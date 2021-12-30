import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

import numpy as np


class GraphMol(Data):
    _MIN_ALLOC = 8

    _edge_alloc: int = 0
    _edge_index: Tensor = None
    _edge_attr: Tensor = None

    def add_edge(self, a: int, b: int, attr: Tensor):
        if self._edge_alloc >= self._edge_index.shape[1]:
            self._edge_index = torch.cat(
                [self._edge_index, torch.empty_like(self._edge_index)], dim=1
            )

            self._edge_attr = torch.cat(
                [self._edge_attr, torch.empty_like(self._edge_attr)], dim=0
            )

        self._edge_index[0, self._edge_alloc] = a
        self._edge_index[1, self._edge_alloc] = b
        self._edge_attr[self._edge_alloc] = attr

        self._edge_alloc += 1

    @property
    def edge_index(self) -> Tensor:
        return self._edge_index[:, : self._edge_alloc]

    @edge_index.setter
    def edge_index(self, value: Tensor):
        size = max(value.shape[1], GraphMol._MIN_ALLOC)

        if self._edge_index is None or size > self._edge_index.shape[1]:
            self._edge_index = torch.empty(
                (2, size), device=value.device, dtype=torch.long
            )

        self._edge_index[:, : value.shape[1]] = value
        self._edge_alloc = value.shape[1]

    @property
    def edge_attr(self) -> Tensor:
        return self._edge_attr[: self._edge_alloc]

    @edge_attr.setter
    def edge_attr(self, value: Tensor):
        size = max(value.shape[0], GraphMol._MIN_ALLOC)

        if self._edge_attr is None or size > self._edge_attr.shape[0]:
            self._edge_attr = torch.empty(
                (size, value.shape[1]), device=value.device, dtype=value.dtype
            )

        self._edge_attr[: value.shape[0]] = value
        self._edge_alloc = value.shape[0]

    def random_trace(self):
        if self.edge_index.shape[1] == 0:
            return []

        start = np.random.choice(self.x.shape[0])
        adj = to_dense_adj(self.edge_index)[0]

        attr_lookup = {}
        for i in range(self.edge_index.shape[1]):
            a, b = self.edge_index[:, i]
            a = int(a)
            b = int(b)
            attr_lookup[(a, b)] = i
            attr_lookup[(b, a)] = i

        # ((focus, neighbors, pick, attr))
        trace = []
        queue = [start]

        while len(queue) > 0:
            focus = queue[-1]

            neighbors = torch.cat(
                [torch.where(adj[focus, :])[0], torch.where(adj[:, focus])[0]]
            )

            if len(neighbors) == 0:
                # Stop.
                queue.pop(-1)
                trace.append((focus, None, None, None))
            else:
                # Produce result
                pick = int(np.random.choice(neighbors.cpu()))

                if not pick in queue:
                    queue.append(pick)

                trace.append(
                    (focus, neighbors, pick, self.edge_attr[attr_lookup[(focus, pick)]])
                )
                adj[focus, pick] = 0
                adj[pick, focus] = 0

        return trace

    def local_distance(self, idx: int, max_dist: int, device=None) -> Tensor:
        r = torch.zeros((self.x.shape[0], max_dist), device=device)

        if self.edge_index.shape[1] == 0:
            return r

        adj = to_dense_adj(self.edge_index)[0]

        closed = set()
        dist = {idx: 0}
        queue = [idx]

        while len(queue) > 0:
            focus = queue.pop(0)
            d = dist[focus]
            if d >= max_dist:
                continue

            r[focus][d] = 1
            closed.add(focus)

            neighbors = torch.cat(
                [torch.where(adj[focus, :])[0], torch.where(adj[:, focus])[0]]
            )

            for n in neighbors:
                n = int(n)
                if not n in queue and not n in closed:
                    queue.append(n)
                    dist[n] = d + 1

        return r
