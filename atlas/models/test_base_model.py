
import pathlib
import tempfile
import unittest

import numpy as np

from .base_model import TestModel


class TestBaseModel(unittest.TestCase):

    def test_create_explicit(self):
        mod = TestModel.create({
            'input_size': 5,
            'hidden_size': 6,
            'output_size': 7
        })

        model = mod.models['model']
        
        self.assertEqual(model[0].in_features, 5)
        self.assertEqual(model[0].out_features, 6)
        self.assertEqual(model[2].in_features, 6)
        self.assertEqual(model[2].out_features, 7)

    def test_create_default_args(self):
        mod = TestModel.create({})

        model = mod.models['model']
        
        self.assertEqual(model[0].in_features, 10)
        self.assertEqual(model[0].out_features, 20)
        self.assertEqual(model[2].in_features, 20)
        self.assertEqual(model[2].out_features, 2)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            mod = TestModel.create({
                'input_size': 5,
                'hidden_size': 6,
                'output_size': 7,
                'foo': 5
            })

    def test_save_load(self):
        mod = TestModel.create({
            'input_size': 5,
            'hidden_size': 6,
            'output_size': 7
        })

        model = mod.models['model']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            model_path = tmp / 'model'

            mod.save(str(model_path))

            other = TestModel.load(str(model_path), device='cpu')
            other_model = other.models['model']

            p1 = next(model.parameters()).detach().numpy()
            p2 = next(other_model.parameters()).detach().numpy()

            self.assertTrue(np.all(p1 == p2))


if __name__ == '__main__':
    unittest.main()
