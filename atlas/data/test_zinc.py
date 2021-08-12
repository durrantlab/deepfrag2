import pathlib
import tempfile
import unittest

from .zinc import ZINCMolProvider

F1 = """\
smiles zinc_id
Cn1cnc2c1c(=O)n(C[C@H](O)CO)c(=O)n2C ZINC000000000221
OC[C@@H]1O[C@H](Oc2ccc(O)cc2)[C@@H](O)[C@H](O)[C@H]1O ZINC000000000964
Cc1cn([C@H]2O[C@@H](CO)[C@H](O)[C@H]2F)c(=O)[nH]c1=O ZINC000000001484
Nc1nc2c(ncn2COC(CO)CO)c(=O)[nH]1 ZINC000000001505
Nc1nc2c(ncn2CCC(CO)CO)c(=O)[nH]1 ZINC000000001899
"""

F2 = """\
smiles zinc_id
NC(=O)CC[C@H](NC(=O)[C@@H]1CCC(=O)N1)C(=O)O ZINC000004899811
C[C@]1(F)[C@@H](O)[C@@H](CO)O[C@@H]1n1ccc(=O)[nH]c1=O ZINC000028470996
C[C@@]1(F)[C@H](O)[C@@H](CO)O[C@@H]1n1ccc(=O)[nH]c1=O ZINC000028471000
"""


class TestZINC(unittest.TestCase):
    def test_zinc_provider(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            p = pathlib.Path(tmpdir)

            with open(p / "ABCD.smi", "w") as f:
                f.write(F1)

            with open(p / "WXYZ.smi", "w") as f:
                f.write(F2)

            z = ZINCMolProvider(tmpdir)
            self.assertEqual(len(z), 8)

            self.assertEqual(
                z[2].iso_smiles, "Cc1cn([C@H]2O[C@@H](CO)[C@H](O)[C@H]2F)c(=O)[nH]c1=O"
            )

            self.assertEqual(
                z[5].iso_smiles, "NC(=O)CC[C@H](NC(=O)[C@@H]1CCC(=O)N1)C(=O)O"
            )


if __name__ == "__main__":
    unittest.main()
