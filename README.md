# 10708_F23
# Geometry-Aware Relational Graph Neural Network  and Language Model for Protein-Protein Interaction Predictions. 

This repository contains the code for the fusion model that we implemented out of a GNN-based model and a ProtBERT-based language model for protein-protein interactions predictions from the Dockground and Marks Dataset.

## Overview

*GeomEtry-Aware Relational Graph Neural Network (**GearNet**)* is a simple yet effective structure-based protein encoder. 
It encodes spatial information by adding different types of sequential or structural edges and then performs relational message passing on protein residue graphs, which can be further enhanced by an edge message passing mechanism.
Though conceptually simple, GearNet augmented with edge message passing can achieve very strong performance on several benchmarks in a supervised setting.

This codebase is based on PyTorch and [TorchDrug] ([TorchProtein](https://torchprotein.ai)). 
It supports training and inference with multiple GPUs.
The documentation and implementation of our methods can be found in the [docs](https://torchdrug.ai/docs/) of TorchDrug.
To adapt our model in your setting, you can follow the step-by-step [tutorials](https://torchprotein.ai/tutorials) in TorchProtein.

[TorchDrug]: https://github.com/DeepGraphLearning/torchdrug

Given below is the link to the Google Drive folders for the datasets Dockground and Marks:

- [PGM_Project](https://drive.google.com/drive/folders/1hyxq0uGVqJnmXYsmbA0sJG01SXpG8Q0i?usp=drive_link)


## Installation

You may install the dependencies via either conda or pip. Generally, GearNet works
with Python 3.7/3.8 and PyTorch version >= 1.8.0.

### From Conda

```bash
conda install torchdrug pytorch=1.8.0 cudatoolkit=11.1 -c milagraph -c pytorch-lts -c pyg -c conda-forge
conda install easydict pyyaml -c conda-forge
```

### From Pip

```bash
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torchdrug
pip install easydict pyyaml
```

### Using Docker

First, make sure to setup docker with GPU support ([guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).

Next, build docker image

```bash
docker build . -t GearNet
```

Then, after image is built, you can run training commands from within docker with following command

```bash
docker run -it -v /path/to/dataset/directory/on/disk:/root/scratch/ --gpus all GearNet bash
```
