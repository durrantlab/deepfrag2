
import argparse
import pathlib

import h5py
from tqdm import tqdm


def append_file(d_smiles: h5py.Dataset, d_zinc: h5py.Dataset, fp: pathlib.Path):
    
    arr_smiles = []
    arr_zinc = []

    with open(fp, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
                
            smiles, zinc = line.split()
            
            arr_smiles.append(smiles)
            arr_zinc.append(zinc)

    prev_size = len(d_smiles)
    new_size = prev_size + len(arr_smiles)

    d_smiles.resize((new_size,))
    d_zinc.resize((new_size,))

    d_smiles[prev_size:] = arr_smiles
    d_zinc[prev_size:] = arr_zinc


def convert(zinc: pathlib.Path, out: pathlib.Path):
    data_files = [x for x in zinc.iterdir() if x.suffix == '.smi']
    print(f'Found {len(data_files)} files.')

    f = h5py.File(str(out), 'w')

    string_type = h5py.string_dtype(encoding='ascii')
    d_smiles = f.create_dataset('smiles', shape=(0,), dtype=string_type, maxshape=(None,))
    d_zinc = f.create_dataset('zinc', shape=(0,), dtype=string_type, maxshape=(None,))

    for fp in tqdm(data_files):
        append_file(d_smiles, d_zinc, fp)

    f.close()
    print('Done!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('zinc', help='Path to zinc directory.')
    parser.add_argument('out', help='Path to output.h5 file.')
    args = parser.parse_args()

    convert(pathlib.Path(args.zinc), pathlib.Path(args.out))

if __name__=='__main__':
    main()
