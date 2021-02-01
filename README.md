# IPCLs-Net: Analyzing Distribution of Intrapapillary Capillary Loops and Predicting Cancer Invasion Depth via Embedding Prior Knowledge of Clustering

# 1. Overview
The morphological pattern of intrapapillary capillary loop (IPCL) under magnification endoscopy directly correlates with the destruction of the original mucosal structure, depth of invasion, and treatment modality chosen for submucosal invasive cancer. Endoscopists have utilized it on diagnosing cancer with only limited clinical value, because the relationship of the distribution of IPCL with the depth of invasion is not quantitative and clear currently. Here we show that the complex distribution of IPCL can be quantified and used to predict the depth of invasion via an IPCLs-Net that is a novel artificial intelligence model embedded with prior knowledge of clustering. We have successfully implemented the IPCLs-Net model on an endoscopic diagnosis system for clinical utilization. Experimental results demonstrate that our model achieves remarkable performance in quantifying IPCL distribution on a dataset containing 14,832 IPCL vessels and significantly improves the diagnostic ability of endoscopists in predicting the depth of invasion. IPCLs-Net is of great significance for endoscopists to know the pathology and depth of invasion in advance, and to make correct treatment modality chosen.

# 2. System requirements
  2.1 Hardware Requirements
    The package requires only a standard computer with GPU and enough RAM to support the operations defined by a user. 
    For optimal performance, we recommend a computer with the following specs:
    RAM: 32+ GB
    CPU: 8+ cores, 3.6+ GHz/core
    GPU：11+ GB (such as GeForce RTX 2080 Ti GPU)
  
  2.2 Software Requirements
  2.2.1 OS Requirements
          This package is supported for Windows operating systems.
  2.2.2 Installing CUDA 10.0 on Windows 10.
  2.2.3 Installing Python 3.6+ on Windows 10.
  2.2.4 Python Package Versions
	    Numpy 1.19.2
	    Pytorch 1.6.0
      Python 3.8.3
	    Seaborn 0.11.1
      Anaconda 4.9.2
	 
# 3. Installation Guide
  A working version of CUDA, python and tensorflow. This should be easy and simple installation. 
  CUDA (https://developer.nvidia.com/cuda-downloads)
  Pytorch (https://pytorch.org/) 
  Python (https://www.python.org/downloads/)
  Anaconda (https://www.anaconda.com/)
  
# 4. Usage of source code
  Enter into folder "IPCLs-Net" for analyzing IPCL distribution.
  Enter into folder "Pathology_prediction_model" for predicting pathology types based on the IPCL distribution.
  
