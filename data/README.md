# MOAD-like data

This directory should contain data formatted as follows:

1. The dataset should residue in its own directory (e.g., `data/moad/`).
2. That directory should contain a `csv` file that divides the data per protein
   family (e.g., `data/moad/every.csv). [Here's an
   example](https://bindingmoad.org/files/csv/every.csv).
3. The directory should contain a subdirectory (any name) with the PDB files.
   The files should be named `*.bio*`.

# Scripts for downloading data

1. Rather than trying to set up the data yourself, you can use the provided
   scripts.
2. `download_moad.sh` will download the entire MOAD database. It's many GB of
   data.
3. `download_moad_little.sh` will download a subset of the MOAD database. It's
   good for testing.