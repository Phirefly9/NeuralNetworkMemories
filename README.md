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

## initial results

### base network with no memory

```
Test set: Average loss: 0.0711, Accuracy: 9771/10000 (98%)
```

### memory on from epoch 1

```
python main.py --epochs 10 --memory_epoch 1
Test set: Average loss: 0.1455, Accuracy: 9576/10000 (96%)
```

### memory on from epoch 3
```
python main.py --epochs 10 --memory_epoch 3
Test set: Average loss: 0.1160, Accuracy: 9658/10000 (97%)
```

### memory on from epoch 5
```
python main.py --epochs 10 --memory_epoch 5
Test set: Average loss: 0.0946, Accuracy: 9716/10000 (97%)
```