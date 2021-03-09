
import unittest

from .molecule_util import MolGraph, SimpleAtomFeaturizer, SimpleBondFeaturizer


class TestMoleculeUtil(unittest.TestCase):

    def test_aspirin_simple(self):
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O' # Aspirin
        m = MolGraph.from_smiles(smiles)

        self.assertEqual(m.atom_coords.shape, (13,3))
        self.assertEqual(m.atom_types.shape, (13,28))
        self.assertEqual(m.bond_index.shape, (2,26))
        self.assertEqual(m.bond_types.shape, (26,5))

    def test_azythromycin_simple(self):
        smiles = 'CCC1C(C(C(N(CC(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O'
        m = MolGraph.from_smiles(smiles)

        self.assertEqual(m.atom_coords.shape, (52,3))
        self.assertEqual(m.atom_types.shape, (52,28))
        self.assertEqual(m.bond_index.shape, (2,108))
        self.assertEqual(m.bond_types.shape, (108,5))

if __name__ == '__main__':
    unittest.main()
