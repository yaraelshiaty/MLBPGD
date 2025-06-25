# multigrid

This repository contains code for the multilevel Bregman proximal gradient descent (MLBPGD) method, with a focus on tomographic reconstruction, deblurring problems, and  D-optimal design in accordance to the numerical examples provided in the corresponding paper.

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
  Example files such as `doptimaldesign_ML.py`, `KLbAx_deblurring_ML.py`, and `KLAxb_Shannon_reconstruction_ML.py` demonstrate how to set up and run experiments. The `_SL.py` files are their single level counterparts
  Running these scripts will generate the `.npz` result files used for plotting and analysis.

## Getting Started

1. **Install dependencies**  
   Make sure you have Python 3.8+ and the required packages (see `requirements.txt` if available).

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