from setuptools import setup, find_packages
import os

# Read requirements from environment.yml pip section
# You might want to create a separate requirements.txt for this
pip_requirements = [
 "numpy==1.23.5",
 "numba==0.56.4",
 "ProDy==2.4.1",
 "torch==1.11.0+cpu",
 "torchvision==0.12.0+cpu",
 "torchaudio==0.11.0+cpu",
 "pytorch-lightning==1.7.1",
 "scipy==1.11.4",
 "pandas==1.4.4",
 "rdkit==2024.3.1",
 "k3d==2.15.2",
 "py3Dmol==2.4.0",
 "h5py==3.7.0",
 "scikit-learn==1.5.1",
 "torchinfo==1.8.0",
 "wandb==0.15.8",
 "torch_geometric",
 "filelock==3.13.1",
 "regex",
 "sacremoses==0.0.53",
 "sentencepiece==0.2.0",
 "tokenizers==0.20.0",
 "torchmetrics==0.11.0",
 "matplotlib==3.8.3",
 "protobuf==3.20.3",
 "Fancy-aggregations",
 "wget",
 "fair-esm",
 "flake8==4.0.1",
 "mypy==1.10.0",
 "pytest==7.1.2",
]
setup(
 name="deepfrag2cpu",
 version="1.0.0",
 author="Your Name",
 author_email="SOME_EMAIL",
 license="MIT",
 description="DeepFrag2CPU - Your package description",
 long_description=open("README.md").read() if os.path.exists("README.md") else "",
 long_description_content_type="text/markdown",
 url="https://github.com/yourusername/deepfrag2cpu",
 packages=['apps', 'collagen', 'molbert', 'transformers'],
 include_package_data=True,
 install_requires=pip_requirements,
 entry_points={
  'console_scripts': [
   'deepfrag2cpu=MainDF2:main',
  ],
 },
 classifiers=[
  "Development Status :: 4 - Beta",
  "Intended Audience :: Developers",
  "Programming Language :: Python :: 3.9",
 ],
 python_requires=">=3.9,<3.10"
)