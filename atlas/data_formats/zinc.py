
import pathlib
from typing import List, Dict

import h5py
from tqdm import tqdm

from ..molecule_util import MolGraphProvider, MolGraph


class ZINCMolGraphProvider(MolGraphProvider):
    """
    A ZINCMolGraphProvider can iterate over a directory containing ZINC
    smiles files.

    For example, the expected directory structure is:
    ```
    zinc/CAAA.smi
    zinc/CAAB.smi
    ...
    ```

    where each file is structured like:
    ```
    smiles zinc_id
    Cn1cnc2c1c(=O)n(C[C@H](O)CO)c(=O)n2C ZINC000000000221
    OC[C@@H]1O[C@H](Oc2ccc(O)cc2)[C@@H](O)[C@H](O)[C@H]1O ZINC000000000964
    Cc1cn([C@H]2O[C@@H](CO)[C@H](O)[C@H]2F)c(=O)[nH]c1=O ZINC000000001484
    Nc1nc2c(ncn2COC(CO)CO)c(=O)[nH]1 ZINC000000001505
    Nc1nc2c(ncn2CCC(CO)CO)c(=O)[nH]1 ZINC000000001899
    ```
    """

    def __init__(self, basedir: str, make_3D: bool = True):
        self.basedir = pathlib.Path(basedir)
        self.make_3D = make_3D

        self.index: Dict[str, List[int]] = self._build_index()
        self.counts: Dict[str, int] = {
            k: len(self.index[k]) for k in self.index}
        self.total: int = sum([self.counts[k] for k in self.counts])

        # file_order represents the canonical ordering for files.
        self.file_order: List[str] = sorted([k for k in self.index])

    def _index_zinc_file(self, fp: pathlib.Path) -> List[int]:
        idx = []

        with open(fp, 'rb') as f:
            for line in f:
                idx.append(f.tell())

        # Ignore last blank newline.
        idx = idx[:-1]

        return idx

    def _build_index(self) -> Dict[str, List[int]]:
        files = list(self.basedir.iterdir())
        
        index = {}

        for fp in tqdm(files, desc='Building ZINC index...'):
            index[fp.stem] = self._index_zinc_file(fp)

        return index

    def __len__(self):
        return self.total

    def __getitem__(self, idx) -> MolGraph:
        assert idx >= 0 and idx < len(self)

        file_index = 0
        while self.counts[self.file_order[file_index]] <= idx:
            idx -= self.counts[self.file_order[file_index]]
            file_index += 1

        fp = self.basedir / f'{self.file_order[file_index]}.smi'

        with open(fp, 'rb') as f:
            f.seek(self.index[self.file_order[file_index]][idx])
            line = f.readline()

            smi, zinc_id = line.decode('ascii').split()

        m = MolGraph.from_smiles(smi, make_3D=self.make_3D)
        m.meta['zinc_id'] = zinc_id

        return m

class ZINCMolGraphProviderH5(MolGraphProvider):
    """
    This version of the ZINCMolGraphProvider iterates over a pre-processed
    h5 format of the ZINC dataset.

    Use the atlas.convert.zinc_to_h5py utility to convert.
    """

    def __init__(self, db: str, make_3D: bool = True, in_mem: bool = False):
        self.fp = h5py.File(db, 'r')
        self.make_3D = make_3D
        self.in_mem = in_mem

        self.d_smiles = self.fp['smiles']
        self.d_zinc = self.fp['zinc']

        if in_mem:
            self.d_smiles = self.d_smiles[()]
            self.d_zinc = self.d_zinc[()]
            self.fp.close()

    def __len__(self):
        return len(self.d_smiles)

    def __getitem__(self, idx) -> MolGraph:
        m = MolGraph.from_smiles(self.d_smiles[idx].decode('ascii'), make_3D=self.make_3D)
        m.meta['zinc_id'] = self.d_zinc[idx].decode('ascii')

        return m
