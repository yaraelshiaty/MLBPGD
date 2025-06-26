# multigrid

This repository contains code for the multilevel Bregman proximal gradient descent (MLBPGD) method, with a focus on tomographic reconstruction, deblurring problems, and D-optimal design in accordance with the numerical examples provided in the corresponding paper.

The codebase supports both research and reproducibility, and is organized for modularity and extensibility. Single-level (= classical Bregman proximal gradient descent) implementations of the examples are provided for comparison.

## Structure

- **multilevel/**  
  Core multilevel optimization framework, including grid operators, coarse correction conditions, and optimization routines.

- **plots/**  
  Scripts and Jupyter notebooks for visualizing results and extracting images from TensorBoard logs.  
  See `plots/README.md` for details.

- **runs/**  
  Output directory for experiment logs and results (e.g., `.npz` files and TensorBoard runs).

- **example scripts**  
  Example files such as `doptimaldesign_ML.py`, `KLbAx_deblurring_ML.py`, and `KLAxb_Shannon_reconstruction_ML.py` demonstrate how to set up and run experiments. The `_SL.py` files are their single level counterparts.
  Running these scripts will generate the `.npz` result files used for plotting and analysis.

## Getting Started

1. **Create and activate the environment**  
   All core dependencies (including `numpy`, `scipy`, and `astra-toolbox`) are managed by conda.  
   Run:
   ```bash
   conda env create -f environment.yml
   conda activate mlbpgd-env
   ```

2. **Run an example**  
   ```bash
   python doptimaldesign_ML.py
   ```
   This will run a D-optimal design experiment and save results in the `runs/` directory.

3. **Visualize results**  
   Use the notebooks in `plots/` to compare multilevel and single-level performance, or extract images from TensorBoard logs.

## Notes

- The `.npz` result files for plotting can be obtained by running the example scripts.
- TensorBoard logging is optional and can be toggled via the `log` hyperparameter in each script.
- For more details on plotting and data extraction, see `plots/README.md`.

## CUDA-dependent Packages

Some dependencies require a CUDA version that matches your system:

- **astra_toolbox**: Installed via conda using the `environment.yml`.
- **torch_scatter**: [Official wheels and instructions](https://pytorch-geometric.com/whl/)

**Steps for torch_scatter:**
1. Check your CUDA version:
   ```bash
   nvcc --version
   ```
   or
   ```bash
   nvidia-smi
   ```

2. Install the compatible version of `torch_scatter` using the command from [https://pytorch-geometric.com/whl/](https://pytorch-geometric.com/whl/) that matches your PyTorch and CUDA version.  
   For example:
   ```bash
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
   ```
   Replace `cu121` with your CUDA version.

**Note:**  
If you do not have a GPU or CUDA, install the CPU-only version of `torch_scatter`.