# pip install pydocstyle

#find . -name "*.py" | grep -v old_apps | grep -v docker | grep -v transformers | grep -v molbert | grep -v tests | grep -v docs | awk '{print "pydocstyle --convention=google --ignore=D407 " $1}'

find . -name "*.py" | grep -v old_apps | grep -v docker | grep -v transformers | grep -v molbert | grep -v tests | grep -v docs | grep -v "zinc.py" | grep -v "pocket_dataset.py" | grep -v "moad_precache_fragments.py" | grep -v "easy_app.py" | grep -v "lr_finder.py" | grep -v "transforms.py" | grep -v "clustered.py" | grep -v "zinc_tsne.py" | grep -v "graph_mol.py" | grep -v "zinc_to_h5py.py" | grep -v "\/test\/" > files.txt

cat files.txt | awk '{print "pydocstyle --ignore=D403,D211,D400,D415,D205,D406,D213,D212,D413,D407 " $1}'

cat files.txt | awk '{print "flake8 " $1}'

cat files.txt | awk '{print "flake8 --select=F401 " $1}'  # specifically for unused imports

rm files.txt
