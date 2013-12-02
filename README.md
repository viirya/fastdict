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

	python fast_binary.py -f bigann_base.bvecs -v bvecs -n 100000000 -k 10 -o 0 -s random -i y -e bigann_100000000.npz

Parameters:

*-f: image feature file
*-v: feature file format
*-n: number of image features to read
*-k: retrieve top-k neighbors
*-o: offset of reading features (begin from offset)
*-s: storage method (dict, redis, random)
*-i: whether runing indexing (y/n), default is 'n'
*-e: indexing file for writing (when -i 'y') and reading (when -i 'n')



