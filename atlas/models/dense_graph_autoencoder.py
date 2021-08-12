import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric

from ..molecule_util import MolGraph
from .model_interfaces import MolAutoencoder


class EdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(EdgeConv, self).__init__(aggr="mean")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.lin0 = torch.nn.Linear(self.in_channels + self.edge_dim, self.out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        merged = torch.cat([x_j, edge_attr], 1)
        msg = self.lin0(merged)

        return msg


class GCNBlock(torch.nn.Module):
    def __init__(self, in_channels, num_layers, layer_size, edge_dim):
        super(GCNBlock, self).__init__()

        self.conv_input = EdgeConv(in_channels, layer_size, edge_dim)
        self.blocks = torch.nn.ModuleList(
            [EdgeConv(layer_size, layer_size, edge_dim) for i in range(num_layers)]
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv_input(x, edge_index, edge_attr)

        for blk in self.blocks:
            z = blk(x, edge_index, edge_attr)
            z = F.relu(z)

            x += z

        return x


class DenseGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, num_layers, layer_size, edge_dim):
        super(DenseGCNBlock, self).__init__()

        self.conv_input = EdgeConv(in_channels, layer_size, edge_dim)
        self.blocks = torch.nn.ModuleList(
            [
                EdgeConv((layer_size * (i + 1)), layer_size, edge_dim)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv_input(x, edge_index, edge_attr)

        outputs = [x]

        for blk in self.blocks:
            h = torch.cat(outputs, 1)
            z = blk(h, edge_index, edge_attr)

            outputs.append(z)
            x = z

        return x


class FPEncoder(torch.nn.Module):
    def __init__(self, in_channels, num_layers, layer_size, edge_dim):
        super(FPEncoder, self).__init__()
        self.gcn = DenseGCNBlock(in_channels, num_layers, layer_size, edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.gcn(x, edge_index, edge_attr)
        x = torch.tanh(x)
        return x


class FPDecoder(torch.nn.Module):
    def __init__(self, out_channels, num_layers, layer_size, edge_dim):
        super(FPDecoder, self).__init__()
        self.gcn = DenseGCNBlock(layer_size, num_layers, layer_size, edge_dim)
        self.lin = nn.Linear(layer_size, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.gcn(x, edge_index, edge_attr)
        x = self.lin(x)
        x = torch.tanh(x)
        return x


class DenseGraphAutoencoder(MolAutoencoder):
    @staticmethod
    def get_params() -> dict:
        return {
            # Size of atom attribute vectors.
            "atom_type_dim": 28,
            # Size of bond attribute vectors.
            "bond_type_dim": 5,
            # Number of dense encoder layers.
            "encoder_layers": 10,
            # Layer/hidden size.
            "z_size": 128,
            # Number of dense decoder layers.
            "decoder_layers": 10,
        }

    @staticmethod
    def build(args: dict) -> dict:
        encoder = FPEncoder(
            in_channels=args["atom_type_dim"],
            num_layers=args["encoder_layers"],
            layer_size=args["z_size"],
            edge_dim=args["bond_type_dim"],
        )

        decoder = FPDecoder(
            out_channels=args["atom_type_dim"],
            num_layers=args["decoder_layers"],
            layer_size=args["z_size"],
            edge_dim=args["bond_type_dim"],
        )

        return {"encoder": encoder, "decoder": decoder}

    def encode(self, mol: MolGraph) -> torch.Tensor:
        z = self.models["encoder"](mol.atom_types, mol.bond_index, mol.bond_types)
        return z

    def decode(self, z: torch.Tensor, template: MolGraph) -> MolGraph:
        p = self.models["decoder"](z, template.bond_index, template.bond_types)

        template.atom_types = p
        return template
