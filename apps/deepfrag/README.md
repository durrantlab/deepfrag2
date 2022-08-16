# DeepFrag

## Steps to use

1. Make sure moad-like data is present in the `../../data/` directory. See the `README.md` file in
   that directory for more information.
2. DeepFrag assumes collagen is available via `import collagen`. If it's not installed globally,
   you can just `ln -s ../../collagen ./` 
3. Make sure you have installed all dependencies. You might try something like this:
   `cd ../../docker/; python3 -m pip install -r requirements.txt.cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html`

