## Overview
This is just a toy example for comparing PyTorch and Mindspore, using unidirectional GRU on a simple-looking task.

Dataset can be obtained from [here](https://drive.google.com/drive/folders/1Amc8jDHDgLTTfB0h0oYovq02JRmkG9Ur?usp=sharing).

To train the model, simply navigate to `src/` and run:

`python run.py --data_dir=DATA_DIR --platform=PLATFORM --device=DEVICE`

where DATA_DIR is the directory where the dataset files are saved, PLATFORM is either PyTorch or Mindspore, and DEVICE is 
either GPU or CPU. Other training settings can also be specified in this way, check `run.py` for details.

## Requirements
* Python 3.7.5
* PyTorch 1.4.0
* Mindspore 1.3.0
* CUDA 10.1

Note that the Mindspore model can only be run under GPU mode (CPU version doesn't work on my Windows). Also setting dropout rate to be greater
than 0 causes the Mindspore model to fail, with some unreadable error messages. It's quite hard to use ...
