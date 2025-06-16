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