from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="deepfrag2cpu",
    version="1.0.1",
    author="Jacob D. Durrant",
    author_email="durrantj@pitt.edu",
    license="MIT",
    description="DeepFrag2CPU - a deep convolutional neural network for lead optimization",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deepfrag2cpu",
    packages=find_packages(),
    py_modules=["MainDF2", "InferenceDF2"],
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'deepfrag2full=MainDF2:main',
            'deepfrag2=InferenceDF2:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9,<3.10",
    extras_require={
        'dev': [
            'flake8==4.0.1',
            'mypy==1.10.0',
            'pytest==7.1.2',
        ]
    }
)
