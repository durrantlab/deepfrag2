
# Data Formats

This page describes the various data interfaces available in the `atlas` module.

# ZINC

[ZINC](https://zinc.docking.org/) is a free database of small-molecule compounds.

The SMILES data is contained in several `.smi` files. The files are structured like:
```
smiles zinc_id
O=C1c2ccccc2C(=O)C(O)(O)C1(O)O ZINC000000481038
O=C1c2no[n+]([O-])c2C(=O)c2c1no[n+]2[O-] ZINC000002290214
O=C1c2no[n+]([O-])c2C(=O)c2no[n+]([O-])c21 ZINC000002765798
```

The directory layout would be something like:
```
zinc/AAAA.smi
zinc/AAAB.smi
zinc/AAAC.smi
...
```

### `ZINCMolGraphProvider`

The `ZINCMolGraphProvider` is an interface to the raw zinc directory. Example usage is as follows:

```py
from atlas.data_formats.zinc import ZINCMolGraphProvider

# If make_3D is set, loaded molecules will have 3D coordinates.
# Note: this makes iteration much slower.
zinc = ZINCMolGraphProvider('./path/to/zinc', make_3D=False)

# Compute number of SMILES strings.
num_examples = len(zinc)

# Load a SMILES string as a MolGraph
g = zinc[0]

print(g.smiles) # "O=C1c2ccccc2C(=O)C(O)(O)C1(O)O"
print(g.meta['zinc_id']) # "ZINC000000481038"
```

During initialization `ZINCMolGraphProvider` needs to build an index of the smiles files which can take a few minutes depending on the size of the ZINC database.

### `ZINCMolGraphProviderH5`

`ZINCMolGraphProviderH5` is a faster way to access ZINC smiles strings but requires a conversion to the h5 format before. To convert a ZINC database, you can use the following tool:

```sh
$ python -m atlas.convert.zinc_to_h5 <path/to/zinc> <out/zinc.h5>
```

The interface is exactly the same as `ZINCMolGraphProvider`:

```py
from atlas.data_formats.zinc import ZINCMolGraphProviderH5

# If make_3D is set, loaded molecules will have 3D coordinates.
# Note: this makes iteration much slower.

# If in_mem is set, the entire ZINC database is loaded into memory.
zinc = ZINCMolGraphProviderH5('./out/zinc.h5', make_3D=False, in_mem=False)

# Compute number of SMILES strings.
num_examples = len(zinc)

# Load a SMILES string as a MolGraph
g = zinc[0]

print(g.smiles) # "O=C1c2ccccc2C(=O)C(O)(O)C1(O)O"
print(g.meta['zinc_id']) # "ZINC000000481038"
```
