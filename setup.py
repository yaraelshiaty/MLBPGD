from setuptools import setup, find_packages
import sys

print(
    "\nNOTE: 'astra_toolbox' and 'torch_scatter' require manual installation to match your CUDA version.\n"
    "Please check your CUDA version and follow the instructions in the README to install these packages.\n",
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
        "numpy==2.3.1",
        "Pillow==11.2.1",
        "scipy==1.16.0",
        "scikit-image",
        "tensorboard==2.18.0",
        "torch==2.4.1",
        "torchvision==0.19.1",
        # "astra_toolbox",  # Install manually
        # "torch_scatter",  # Install manually
    ],
    python_requires=">=3.8",
)