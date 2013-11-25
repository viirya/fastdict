### Fast hamming distance-based image retrieval using cuda

This repo includes tools used for conducting experiments of image retrieval by using cuda.

## Prerequisites

Install few python modules:

	pip install bitarray
	pip install redis
	
Checkout and install pycuda.

Download and compile [yael library](https://gforge.inria.fr/projects/yael). Copy generated `yael.py`, `_yael.so` under the same path of these scripts.

This repo uses modified [LSHash](https://github.com/kayzh/LSHash).

## Useful datasets

[ANN_SIFT1B](http://corpus-texmex.irisa.fr/)


## Usage


# Binary code indexing and retrieval using cuda

	python fast_binary.py -f sift_base.fvecs

