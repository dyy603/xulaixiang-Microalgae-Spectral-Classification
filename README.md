# Microalgae Spectral Identification System
ANMM: An Efficient Deep Learning model for Microalgae Spectral Classification
# Project Overview 
The project aims to construct an efficient Microalgae Spectral classification model using deep learning techniques. By utilizing an improved AlexNet network architecture combined with attention mechanisms (multi head self attention mechanism and multi head late attention mechanism), it achieves automatic recognition of common Microalgae Spectral features, providing a high-precision and lightweight solution for real-time monitoring and classification of microalgae.
# Project Structure
```
ANMM/
├── ANMM.pth                   # weight file
├── train1.py                  # Main script for model training
├── model.py                   # Definition of the improved AlexNet model
├── mla.py                     # Implementation of the ANMM attention mechanism
├── msa.py                     # Implementation of the ANMM attention mechanism
├── predict1.py                # Model prediction script
└── dataset/
    ├── train/                 # Training set
    ├── val/                   # Validation set
    └── test/                  # Test set
```
# Core Technologies 
__1.Improved AlexNet__：Add a layer of convolution to enhance the depth of feature extraction.  
__2.Integrate Attention Mechanisms：__
  multi head self attention mechanism and multi head late attention mechanism  
  __3.The early stop mechanism__: Terminate training prematurely when validation set performance stops improving
# Recommended Environment
Python 3.12  
PyTorch == 2.3
# Dataset acquisition and structure
The dataset should be organized in the following structure:
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
├── ├──class1/
│   ├── class2/
│   └── ...
├── test/
├── ├──class1/
│   ├── class2/
│   └── ...
```
Each category folder contains microalgae spectral images corresponding to that category.
# Model Training 
Use the `train1.py` script for model training，after changing the path:
```
python MLSA_train.py
```
