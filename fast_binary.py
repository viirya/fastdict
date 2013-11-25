
import numpy
import yutils
import yael
import timeit
import time
import argparse

from lshash import LSHash


def main():

    parser = argparse.ArgumentParser(description = 'Tools for hamming distance-based image retrieval by cuda')
    parser.add_argument('-f', help = 'The filename of image raw features (SIFT).')
    parser.add_argument('-v', default = 'fvecs', help = 'The format of image raw features.')
    parser.add_argument('-d', default = '128', help = 'Dimensions of raw image feature.')
    parser.add_argument('-o', default = '0', help = 'Offset of accessing raw image features.')
    parser.add_argument('-n', default = '1', help = 'Number of raw image features to read.')
    parser.add_argument('-i', default = 'n', help = 'Whether to perform indexing step.')
    parser.add_argument('-k', default = '10', help = 'Number of retrieved images.')

    args = parser.parse_args()

    d = int(args.d)
    nuse = int(args.n)
    off = int(args.o)

    (feature_vecs, n) = yutils.load_vectors_fmt(args.f, args.v, d, nuse, off, verbose = True)

    np_feature_vecs = yael.fvec_to_numpy(feature_vecs, nuse * d)
    np_feature_vecs = np_feature_vecs.reshape((nuse, d))

    lsh = LSHash(64, d, 1, storage_config = 'redis', matrices_filename = 'project_plane.npz')

    if args.i == 'y':
        index = 0
        for vec in np_feature_vecs:
            lsh.index(vec, extra_data = 'vec' + str(index))
            index += 1

    retrived = lsh.query(np_feature_vecs[0], num_results = int(args.k), distance_func = 'hamming')
    print retrived

if __name__ == "__main__":
    main()

