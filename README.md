LiDAR Scan Project

This repository contains tools for working with [LiDAR scans](https://drive.google.com/drive/u/1/folders/1DliNpAwRJkojgTkAah4oK116x1sy3TcZ).

---

## Requirements

- **VS Code** (or any text editor/IDE of your choice)  
- **Git**  
- **Python 3.12.0**  
- *(if using VS Code)*  
  - Python extension  
  - Code Runner extension  

---

## Setup Instructions

Clone this repository and install the required dependencies.  
All commands below should be run from the **project root**:

```bash
# Install core dependencies
pip install numpy

# Install PyTorch (CPU-only version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric dependencies
pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install --no-cache-dir torch-spline-conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# Install PyTorch Geometric itself
pip install torch-geometric

# Install additional scientific libraries
python -m pip install scipy



find main.py in training_model dir and run code. 
find model_inferencing dir and create a new folder "cleaned_scans" and add new scans or scans not used in training.
go to inferencing.py in model_inference dir and run file to create inferences.
```


# Cross-Platform-Inference-Benchmarking
Evaluating the same LiDAR model inferences across Cloud, Edge, and Mobile devices.


easier to go through documentation and the repo at the same time. it goes through each file, function, and purpose in a neat order!
