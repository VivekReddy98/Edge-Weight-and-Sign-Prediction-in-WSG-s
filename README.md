## Edge Weight and Sign Prediction in WSGs using Weighted Signed Graph Convolutional Network
Graph Data Mining Capstone Project

1) Edge Weight and Sign Prediction using Node Embeddings generated from a Weighted Signed Graph Convolutional Network.
2) A MLP is used for Weight ans Sign Prediction. 
3) WSGN was written using Tensorflow Low-Level API and is vectorized and works for any arbitrary batch size as long as the NxN Matrix fits in the memory (#N : Number of Nodes)
4) MLP's were written in Keras.
5) Modified Signed Spectral Embeddings are given as the initial embeddings for WSGCN.

### Infrastructure Required:
1) Python 3
2) Tensorflow v1.15
3) CUDA enabled GPU (Preferably)
4) Other Python Packages as specified in requirements.txt

### Installation Guide
1) Preferably Install Anaconda.
2) Create a New Environment, cmd: conda create -n myenv python=3.6
3) Switch to the Environment using conda activate myenv
4) Install Tensorflow using cmd: conda install tensorflow-gpu=1.15
5) To install TF from source:
https://gist.github.com/Brainiarc7/6d6c3f23ea057775b72c52817759b25c


## Folder Structure
1) Trained Node Embeddings are found in embeddings folder. Embeddings_[dataset name][numLayers].h5
2) Training History, Results, Loss computed for WSGCN training is found in the results folder.
3) Look into src folder for source code. 
4) exec.py is the entry point to run the WSGCN, edit the specified column for hyperparameters.
5) metrics.py is the entry point to run the MLP's.

## Source Code Walkthrough:
1) generatorUtils.py: Generator utils to clean the dataset, statified splitting, Generate SSE and generate node pairs for the objective function.
2) weightsSGCN.py: The WSGCN weight initialization procedure is handled by this file. (Glorot initialization is used by default).
3) sgcnLayers.py: Completely Vectorized, Layer Definitions. Works for any arbitrary batch size. 
4) sgcn.py: Stacking Layers for forward pass, BackPropoagation and all the required types of losses defined. Placeholder Initilzation for dynamically changing inputs.



