from setuptools import setup, find_packages

setup(
    name="MLBPGD",
    version="0.1.0",
    description="Multilevel Bregman Proximal Gradient Descent",
    author="Yara Elshiaty",
    packages=find_packages(),
    install_requires=[
        "astra_toolbox==2.1.0",
        "matplotlib==3.8.4",
        "numpy==2.3.1",
        "Pillow==11.2.1",
        "scipy==1.16.0",
        "skimage==0.0",
        "tensorboard==2.18.0",
        "torch==2.4.1",
        "torch_scatter==2.0.8",
        "torchvision==0.19.1",
    ],
    python_requires=">=3.8",
)