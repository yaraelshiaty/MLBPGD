# Plots and Data Extraction

This folder contains scripts and notebooks for plotting and extracting data from experiments.

## Extracting Images from TensorBoard

Use `extract_tensorboard_images.py` to extract all images saved to TensorBoard during training.

**Usage:**
```bash
python extract_tensorboard_images.py
```
Edit the `log_dir` variable in the script to point to your run folder.

Extracted images will be saved in a subfolder called `extracted_images` inside your run directory.

## Plotting Multilevel vs Single Level Results

The notebook `plots_Ml.ipynb` loads `.npz` results from both multilevel (ML) and single level (SL) experiments and visualizes the normalized function value versus cumulative CPU time.  
It highlights ML iterations, SL iterations, and marks with a red "x" where the coarse correction (CC) condition fails during ML optimization.  
You can adjust the file paths in the notebook to point to your own experiment results.

**Note:**  
The `.npz` result files can be obtained by running the example files such as `doptimaldesign_ML.py`, `KLbAx_deblurring_ML.py`, etc. These scripts will generate the necessary `.npz` files for