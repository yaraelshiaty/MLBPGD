from setuptools import setup, find_packages
import sys

print(
    "\nNOTE: Core dependencies (numpy, scipy, astra-toolbox, etc.) are installed via conda using environment.yml.\n"
    "Please create your environment with:\n"
    "  conda env create -f environment.yml\n"
    "  conda activate mlbpgd-env\n"
    "After activation, install 'torch_scatter' manually as per your CUDA version (see README).\n",
    file=sys.stderr
)

setup(
    name="MLBPGD",
    version="0.1.0",
    description="Multilevel Bregman Proximal Gradient Descent",
    author="Yara Elshiaty",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.8.4",
        "Pillow==11.2.1",
        "scikit-image",
        "tensorboard==2.18.0",
        "torch==2.4.1",
        "torchvision==0.19.1",
        # numpy, scipy, astra-toolbox installed via conda
        # "torch_scatter",  # Install manually as per CUDA version
    ],
    python_requires=">=3.8",
)