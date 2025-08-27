link to lidar scans:
https://drive.google.com/drive/u/1/folders/1DliNpAwRJkojgTkAah4oK116x1sy3TcZ


requirements
Vs code (any text editor or IDE works of course)
git
Python 3.12.0
Python and Code runner Extensions (if you're using vs code)

run in terminal in project root:
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install --no-cache-dir torch-spline-conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install torch-geometric
python -m pip install scipy


find main.py in training_model dir and run code. 
find model_inferencing dir and create a new folder "cleaned_scans" and add new scans or scans not used in training.
go to inferencing.py in model_inference dir and run file to create inferences.



# Cross-Platform-Inference-Benchmarking
Evaluating the same LiDAR model inferences across Cloud, Edge, and Mobile devices.


easier to go through documentation and the repo at the same time. it goes through each file, function, and purpose in a neat order!
