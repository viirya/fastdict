## Fast hamming distance-based image retrieval using cuda

This repo includes tools used for conducting experiments of image retrieval by using CUDA.

### Prerequisites

Install Python 2.7.

Install Python modules:

- `pip install numpy`
- `pip install bitarray`
- `pip install redis`
	
Other prerequisites:

* PyCuda
* Boost
* Boost::Python
* Boost::Serialization

Download and compile [yael library](https://gforge.inria.fr/projects/yael) with the `--enable-numpy` option. Copy generated `yael.py`, `_yael.so` under the same path of these scripts.


This repo uses modified [LSHash](https://github.com/kayzh/LSHash).

### Build

To build fastdict:

    mkdir build
    cd build
    cmake path_to_fastdict/
    make
    cp fastdict.so path_to_repo/

### Useful datasets

[ANN_SIFT1B](http://corpus-texmex.irisa.fr/)


### Usage


#### Binary code indexing and retrieval using cuda

	python fast_binary.py -f sift_base.fvecs

	python fast_binary.py -f bigann_base.bvecs -v bvecs -n 100000000 -k 10 -o 0 -s random -i y -e bigann_100000000.npz

    python fast_binary_for_indexonly.py -f ../bigann_base.bvecs -v bvecs -n 500000000 -k 10 -o 0 -s random -i y -e bigann_500000000_random_k8_b64 -r 8

Parameters:

* -f: image feature file
* -v: feature file format
* -n: number of image features to read
* -k: retrieve top-k neighbors
* -o: offset of reading features (begin from offset)
* -s: storage method (dict, redis, random)
* -i: whether runing indexing (y/n), default is 'n'
* -e: indexing file for writing (when -i 'y') and reading (when -i 'n')
* -c: whether to perform dict compressing, default is 'n'
* -r: the number of sampled dimensions
* -q: performing sequential sampling, default is 'n' (meaning that default is ramdom sampling)
* -p: querying in compressed domain, default is 'n' (meaning that plain querying mode)
* -g: 'y' for GPU-based uncompression. 'n' for CPU-based.
* -l: 'y' for VLQ base64 mode. default is 'n'
* -b: the level of bucket expansion
* -t: the type of FastDict component. It can be 'int32', 'int8' or 'string'. default is 'int32'
* -u: 'net' or 'local'; how the cuda computation engine is called.
* -host: when -u is 'net', indicating the cuda server location.
* -title: the title string that will be logged at cuda server.
* -gt: the feature file of ground truth.

#### R script to calculate theoretical compression performance

    R --slave --args <binary code length> <number of binary codes> <bit width of bit counts> <number of sampled dimensions> <weight of worst-case> <weight of best-case> < cal_compress_effect.R

For example:

    R --slave --args 64 1000000000 32 32 0.19 0.81 < cal_compress_effect.R




