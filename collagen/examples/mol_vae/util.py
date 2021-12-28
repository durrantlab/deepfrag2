import tempfile

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.data import Data

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from PIL import Image
import numpy as np

from ...core import GraphMol, Mol


slots = [
    (6, 4, 0),  # C
    (7, 2, -1),  # N-
    (7, 3, 0),  # N
    (7, 4, +1),  # N+
    (8, 1, -1),  # O-
    (8, 2, 0),  # O
    (9, 1, 0),  # F
    (17, 1, 0),  # Br
    (35, 1, 0),  # Cl
    (53, 1, 0),  # I
    (16, 2, 0),  # S2
    (16, 4, 0),  # S4
    (16, 6, 0),  # S6
]


def get_atom_slot(atom):
    num = atom.GetAtomicNum()
    val = atom.GetTotalValence()
    charge = atom.GetFormalCharge()

    ent = (num, val, charge)
    if ent in slots:
        return slots.index(ent)
    else:
        return -1


def get_bond_slot(bond):
    order = bond.GetBondTypeAsDouble()
    if order == 2.0:
        return 1
    elif order == 3.0:
        return 2
    else:
        return 0


def mol_to_graph(mol: Mol) -> GraphMol:
    rdmol = mol.rdmol

    Chem.Kekulize(rdmol, clearAromaticFlags=True)

    atom_attr = np.array([get_atom_slot(x) for x in rdmol.GetAtoms()])
    valid = list(np.where(atom_attr > -1)[0])

    edges = []
    edge_attr = []
    for i in range(rdmol.GetNumBonds()):
        bond = rdmol.GetBondWithIdx(i)
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        if a in valid and b in valid:
            edges.append((valid.index(a), valid.index(b)))
            edge_attr.append(get_bond_slot(bond))
        else:
            print("invalid edge")

    x = F.one_hot(torch.tensor(atom_attr[valid]).long(), len(slots)).float()
    edge_index = torch.tensor(edges).reshape((-1, 2)).T.long()
    edge_attr = F.one_hot(torch.tensor(edge_attr).long(), 3).float()  # TODO: bond dim

    return GraphMol(x=x, edge_index=edge_index, edge_attr=edge_attr)


def graph_to_mol(G: GraphMol) -> Mol:
    G = G.to("cpu")
    m = Chem.RWMol()

    if len(G.x) > 0:
        atoms = torch.argmax(G.x, dim=1).numpy()
        for i in range(len(atoms)):
            num, valence, charge = slots[atoms[i]]
            a = Chem.Atom(num)
            a.SetFormalCharge(charge)
            m.AddAtom(a)

        edges = [(int(a), int(b)) for a, b in G.edge_index.T]
        bond_order = torch.argmax(G.edge_attr, dim=1).numpy()
        for i in range(len(edges)):
            a, b = edges[i]
            order = bond_order[i]
            typ = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE][
                order
            ]

            m.AddBond(a, b)
            m.GetBondWithIdx(i).SetBondType(typ)

    return Mol.from_rdkit(m, strict=False)


def sample(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def kld_loss(mu, logvar):
    return torch.mean(
        -0.5 * torch.sum(1 + logvar - (mu ** 2) - torch.exp(logvar), dim=1)
    )


def categorical(vals, size, device=None):
    z = torch.zeros((size,), device=device)
    for v in vals:
        z[v] = 1
    return z


def image_grid(mols, w=5, h=5):
    canvas = Image.new("RGB", (w * 200, h * 200))

    for i in range(len(mols)):
        with tempfile.NamedTemporaryFile() as f:
            try:
                Draw.MolToImageFile(mols[i], f.name + ".png", size=(200, 200))
                im = Image.open(f.name + ".png")
                canvas.paste(im, (((i % w) * 200), (i // w) * 200))
            except:
                pass

    return canvas
