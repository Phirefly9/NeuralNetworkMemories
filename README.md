# Neural Network with sample memory

experiments repo for adding sample memory to a neural network.  The network stores features from previously examined samples up to a threshold amount before it begins to "forget" samples.

Samples are stored and retrieved using the [Facebooks Faiss](https://github.com/facebookresearch/faiss) library

# installation and running

The easiest installation is with using anaconda

## install requirements
```
conda create -yn nnmemories python=3
conda env update -f requirements.yml -n nnmemories
```

## Run code

```
conda activate nnmemories
python main.py
```