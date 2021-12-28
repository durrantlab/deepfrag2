import tempfile
import traceback
from typing import Tuple, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import subgraph

from rdkit.Chem import Draw


import numpy as np
import pytorch_lightning as pl
import wandb


from ...core import Mol, GraphMol
from . import util


class CustomGraphConv(MessagePassing):
    def __init__(
        self,
        latent_size: int,
        num_layers: int,
        edge_size: int,
        aggr: str = "add",
        bias: bool = True,
        **kwargs
    ):
        super(CustomGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.latent_size = latent_size
        self.num_layers = num_layers

        self.weight = nn.Parameter(
            torch.Tensor(num_layers, edge_size, latent_size, latent_size)
        )
        self.rnn = torch.nn.GRUCell(latent_size, latent_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform(self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index: Adj, edge_attr):
        # Make undirected.
        edge_index = edge_index.repeat((1, 2))
        edge_attr = edge_attr.repeat((2, 1))

        bond_idx = (
            torch.argmax(edge_attr, dim=1)
            if len(edge_attr) > 0
            else torch.zeros((0,)).long()
        )

        m = x
        for i in range(self.num_layers):
            m = self.propagate(edge_index, n=i, x=m, bond_idx=bond_idx)
            m = self.rnn(m, x)
            m = F.relu(m)

        return m

    def message(self, n, x_j, bond_idx):
        mat = self.weight[n][bond_idx]
        return torch.bmm(mat, x_j.view((-1, self.latent_size, 1))).view(
            (-1, self.latent_size)
        )


class Encoder(nn.Module):
    def __init__(self, in_size: int, latent_size: int, steps: int, edge_size: int):
        super(Encoder, self).__init__()
        self.in_size = in_size
        self.latent_size = latent_size

        self.gnn = CustomGraphConv(latent_size, num_layers=steps, edge_size=edge_size)
        self.lin = nn.Linear(in_size, latent_size)

        self.fc_mu = nn.Linear(latent_size, latent_size)
        self.fc_logvar = nn.Linear(latent_size, latent_size)

    def forward(self, G):
        x = self.lin(G.x)
        x = F.relu(x)
        x = self.gnn(x, G.edge_index, G.edge_attr)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


def cyclic_kl(warm: int, ramp: int, hold: int, max: float):
    for i in range(warm):
        yield 0
    while True:
        for i in range(ramp):
            yield (float(i) / ramp) * max
        for i in range(hold):
            yield max


class MolVAE(pl.LightningModule):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        z_size: int,
        z_select: int,
        z_bond: int,
        z_atom: int,
        dist_r: int,
        enc_steps: int,
        dec_steps: int,
        use_argmax: bool,
        kl_warm: int,
        kl_ramp: int,
        kl_hold: int,
        kl_max: float,
        image_freq: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.h_size = self.hparams["atom_dim"] + self.hparams["z_size"]
        self.phi_size = 1 + self.hparams["dist_r"] + (self.h_size * 4)

        # Graph embedding.
        self.encoder = Encoder(
            in_size=self.hparams["atom_dim"],
            latent_size=self.hparams["z_size"],
            steps=self.hparams["enc_steps"],
            edge_size=self.hparams["bond_dim"],
        )

        # Internal state update during graph generation.
        self.decoder = CustomGraphConv(
            latent_size=self.h_size,
            num_layers=self.hparams["dec_steps"],
            edge_size=self.hparams["bond_dim"],
        )

        # Graph generation edge selection.
        self.f_select = nn.Sequential(
            nn.Linear(self.phi_size, self.hparams["z_select"]),
            nn.ReLU(),
            nn.Linear(self.hparams["z_select"], 1),
        )

        # Bond selection.
        self.f_bond = nn.Sequential(
            nn.Linear(self.phi_size, self.hparams["z_bond"]),
            nn.ReLU(),
            nn.Linear(self.hparams["z_bond"], self.hparams["bond_dim"]),
        )

        # Atom selection.
        self.f_atom = nn.Sequential(
            nn.Linear(self.hparams["z_size"], self.hparams["z_atom"]),
            nn.ReLU(),
            nn.Linear(self.hparams["z_atom"], self.hparams["atom_dim"]),
        )

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

        self.kl_schedule = cyclic_kl(
            self.hparams["kl_warm"],
            self.hparams["kl_ramp"],
            self.hparams["kl_hold"],
            self.hparams["kl_max"],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_atom_types(self, z_init: Tensor) -> Tensor:
        scores = F.softmax(self.f_atom(z_init), dim=1)
        tau_init = F.one_hot(torch.argmax(scores, 1), self.hparams["atom_dim"])

        return (tau_init, scores)

    def predict_edge(
        self, phi: Tensor, mask: Tensor, train: bool = False
    ) -> Optional[int]:
        edge_mask = torch.cat(
            [torch.tensor(1, device=self.device).reshape((1,)), mask], axis=0
        )

        raw = self.f_select(phi).reshape((-1,))
        scores = F.softmax(raw, dim=0) * edge_mask
        norm = scores / torch.sum(scores)

        if self.hparams["use_argmax"]:
            target = torch.argmax(norm)
        else:
            target = np.random.choice(len(norm), p=norm.cpu().detach().numpy())

        return (None if target == 0 else (target - 1)), F.sigmoid(raw)

    def predict_bond(
        self, phi: Tensor, avail_valency: dict, focus: int, target: int
    ) -> int:
        bond_max = min(avail_valency[focus], avail_valency[target])
        bond_mask = torch.tensor(
            [1, int(bond_max >= 2), int(bond_max >= 3)], device=self.device
        )

        raw = self.f_bond(phi[target]).reshape((-1,))
        scores = F.softmax(raw, dim=0)
        masked = scores * bond_mask
        norm = masked / torch.sum(masked)

        if self.hparams["use_argmax"]:
            pick = torch.argmax(norm)
        else:
            pick = np.random.choice(len(norm), p=norm.cpu().detach().numpy())

        bond_pred = F.one_hot(torch.tensor([pick]), self.hparams["bond_dim"])

        return bond_pred, scores

    def build_phi(self, G: GraphMol, t: int, u: int, H_0: Tensor, N: int) -> Tensor:
        """
        Build phi = [t, D_uv, S_u, S_v, H_0, H_t] for all v in G

        Output has size [N+1, PHI_SIZE]
        (row 0 represents "stop" connection)
        """
        Z = self.h_size
        R = self.hparams["dist_r"]

        phi_H_0 = H_0.reshape((1, Z)).expand((N + 1, Z))
        phi_H_t = torch.mean(G.x, axis=0).reshape((1, Z)).expand((N + 1, Z))
        phi_t = torch.tensor(t, device=self.device).expand(N + 1).reshape((N + 1, 1))
        s_u = G.x[u].reshape((1, Z)).expand((N + 1, Z))
        s_v = torch.cat([torch.zeros((1, Z), device=self.device), G.x], axis=0)
        phi_r = torch.cat(
            [
                torch.zeros((1, R), device=self.device),
                G.local_distance(u, R, device=self.device),
            ],
            axis=0,
        )

        phi = [phi_t, phi_r, s_u, s_v, phi_H_0, phi_H_t]
        phi = torch.cat(phi, axis=1)
        return phi

    def encode(self, G: GraphMol) -> Tuple[Tensor, Tensor]:
        """Encode a mol and return z_mu and z_logvar."""
        z_mu, z_logvar = self.encoder(G)
        return (z_mu, z_logvar)

    def _block_mask(self, mask: Tensor, idx: int):
        mask[idx, :] = 0
        mask[:, idx] = 0

    def decode(self, z_init: Tensor, train_ref: Optional[GraphMol] = None):
        is_train = train_ref is not None
        if is_train:
            loss = {"atom": 0, "bond": 0, "select": 0}
            acc = {"atom": 0, "bond": 0, "select": 0}
            trace = train_ref.random_trace()
            num_edge = 0
            num_bond = 0

        N = len(z_init)

        start = np.random.choice(len(z_init))
        if is_train:
            start = trace[0][0]

        queue = [start]
        visited = set([start])

        atom_types, atom_scores = self.predict_atom_types(z_init)
        if is_train:
            loss["atom"] = self.mse(
                torch.sum(atom_scores, dim=0), torch.sum(train_ref.x, dim=0)
            )
            acc["atom"] = torch.mean(
                (
                    torch.argmax(atom_types, dim=1) == torch.argmax(train_ref.x, dim=1)
                ).float()
            )

            # Replace atom labels with correct types during training.
            atom_types = train_ref.x

        h_init = torch.cat([z_init, atom_types], axis=1)

        # Keep track of remaining "available valency"
        # TODO: better lookup
        avail_valency = [util.slots[int(x)][1] for x in torch.argmax(atom_types, dim=1)]

        H_0 = torch.mean(h_init, axis=0)

        G = GraphMol(
            x=h_init,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, self.hparams["bond_dim"]), dtype=torch.float),
        )
        G.to(self.device)

        # A connection between u and v is valid if mask[u,v] == 1.
        mask = torch.ones((N, N), device=self.device) - torch.eye(N, device=self.device)

        t = 0
        while ((not is_train) and len(queue) > 0) or (is_train and len(trace) > 0):
            if is_train:
                # For a training run, we follow a fixed path specified by trace.
                focus, neighbors, pick, attr = trace.pop(0)
                queue = [focus]

            focus = queue[-1]

            if avail_valency[focus] <= 0 and not is_train:
                self._block_mask(mask, focus)
                queue.pop(-1)
                continue

            phi = self.build_phi(G, t, focus, H_0, N)

            target, edge_scores = self.predict_edge(phi, mask[focus])
            if is_train:
                pred_target = target
                target = pick

            if target is None:
                # Stop for this atom.
                self._block_mask(mask, focus)
                queue.pop(-1)

                if is_train:
                    loss["select"] += self.bce(
                        edge_scores, util.categorical([0], N + 1, device=self.device)
                    )
                    acc["select"] += int(pred_target == target)
                    num_edge += 1
            else:
                # New edge.
                mask[focus, target] = 0
                mask[target, focus] = 0
                visited.add(target)

                if is_train:
                    loss["select"] += self.bce(
                        edge_scores,
                        util.categorical(
                            [x + 1 for x in neighbors], N + 1, device=self.device
                        ),
                    )
                    acc["select"] += int(
                        pred_target is not None and pred_target in neighbors
                    )
                    num_edge += 1

                # Predict bond type.
                bond_pred, bond_scores = self.predict_bond(
                    phi, avail_valency, focus, target
                )

                if is_train:
                    loss["bond"] += self.bce(bond_scores, attr)
                    acc["bond"] += (
                        torch.argmax(bond_scores) == torch.argmax(attr)
                    ).float()
                    num_bond += 1

                    bond_pred = attr

                # Add new edge to graph.
                G.add_edge(focus, target, bond_pred)

                # Valency checks
                avail_valency[focus] -= torch.argmax(bond_pred) + 1
                avail_valency[target] -= torch.argmax(bond_pred) + 1

                if avail_valency[focus] <= 0:
                    self._block_mask(mask, focus)
                    queue.pop(-1)
                if avail_valency[target] <= 0:
                    self._block_mask(mask, target)
                elif not target in queue:
                    queue.append(target)

                # Update state.
                G.x = self.decoder(
                    x=h_init, edge_index=G.edge_index, edge_attr=G.edge_attr
                )

            t += 1

        valid = sorted(list(visited))

        if G.edge_index.shape[1] == 0:
            res = GraphMol(
                x=torch.empty((0, self.hparams["atom_dim"])),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, self.hparams["bond_dim"])),
            )
        else:
            edge_index, edge_attr = subgraph(
                valid,
                edge_index=G.edge_index,
                edge_attr=G.edge_attr,
                relabel_nodes=True,
            )
            res = GraphMol(
                x=atom_types[valid], edge_index=edge_index, edge_attr=edge_attr
            )

        if is_train:
            if num_edge > 0:
                loss["select"] /= num_edge
                acc["select"] /= num_edge

            if num_bond > 0:
                loss["bond"] /= num_bond
                acc["bond"] /= num_bond

            return (res, loss, acc)
        else:
            return res

    def log_image_near(self, mol):
        G = util.mol_to_graph(mol)
        G.to(self.device)
        z_mu, z_logvar = self.encode(G)

        mols = [mol.rdmol]

        for _ in range(24):
            z = util.sample(z_mu, z_logvar)

            G = self.decode(z)
            m = util.graph_to_mol(G)
            mols.append(m.rdmol)

        return util.image_grid(mols)

    def training_step(self, batch, batch_idx):
        loss_select = 0
        loss_atom = 0
        loss_bond = 0
        acc_select = 0
        acc_atom = 0
        acc_bond = 0

        loss_latent = 0

        for mol in batch:
            try:
                G = util.mol_to_graph(mol)
                G.to(self.device)
                z_mu, z_logvar = self.encode(G)
                z = util.sample(z_mu, z_logvar)

                loss_latent += util.kld_loss(z_mu, z_logvar)

                _, loss, acc = self.decode(z, train_ref=G)
                loss_select += loss["select"]
                loss_atom += loss["atom"]
                loss_bond += loss["bond"]
                acc_select += acc["select"]
                acc_atom += acc["atom"]
                acc_bond += acc["bond"]
            except Exception as e:
                print("fail")
                print(traceback.format_exc())

        loss_select /= len(batch)
        loss_atom /= len(batch)
        loss_bond /= len(batch)
        acc_select /= len(batch)
        acc_atom /= len(batch)
        acc_bond /= len(batch)
        loss_latent /= len(batch)

        re_loss = loss_select + loss_atom + loss_bond

        kl_factor = next(self.kl_schedule)
        loss = (kl_factor * loss_latent) + re_loss

        self.log("loss", loss)

        self.log("loss_select", loss_select, prog_bar=True)
        self.log("loss_atom", loss_atom, prog_bar=True)
        self.log("loss_bond", loss_bond, prog_bar=True)
        self.log("loss_latent", loss_latent, prog_bar=True)

        self.log("acc_select", acc_select)
        self.log("acc_atom", acc_atom)
        self.log("acc_bond", acc_bond)

        self.log("kl_factor", kl_factor)

        if batch_idx % self.hparams["image_freq"] == 0:
            canvas = self.log_image_near(batch[0])
            self.logger.experiment.log({"render": wandb.Image(canvas)})

        return loss
