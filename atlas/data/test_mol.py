import unittest

import rdkit.Chem.AllChem as Chem

from .mol import Mol


class TestMol(unittest.TestCase):
    def test_load_smiles(self):
        test = [("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 206.13), ("*CCCC", 57.07)]

        for smi, mass in test:
            m = Mol.from_smiles(smi)
            self.assertAlmostEqual(m.mass, mass, 2)

    def test_from_rdkit(self):
        mol = Chem.MolFromSmiles("CCCC")
        m = Mol.from_rdkit(mol)

        self.assertEqual(m.smiles, "CCCC")
