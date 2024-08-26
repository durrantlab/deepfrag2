import unittest
from collagen import Mol


class TestMol(unittest.TestCase):
    
    def test_fromSmiles(self):
        m1 = Mol.from_smiles('CCCC')
        self.assertEqual(m1.num_atoms, 4)


if __name__=='__main__':
    unittest.main()
