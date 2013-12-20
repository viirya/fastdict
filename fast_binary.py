
import sys
import numpy
import yutils
import yael
import timeit
import time
import os
import argparse

from lshash import LSHash

def load_features(filename, file_format, total_nuse, dimension, lsh, index_folder, offset = 0, run_index = 'n'):

    np_feature_vecs = None
    actual_total_nuse = 0

    for feature_idx_begin in range(offset, total_nuse + offset, 10000000):

        print "loading from " + str(feature_idx_begin)

        nuse = 0
        if (total_nuse + offset) > (feature_idx_begin + 10000000):
            nuse = 10000000
        else:
            nuse = (total_nuse + offset) - feature_idx_begin

        (feature_vecs, actual_nuse) = yutils.load_vectors_fmt(filename, file_format, dimension, nuse, feature_idx_begin , verbose = True)

        part_np_feature_vecs = None

        if file_format == 'fvecs':
            part_np_feature_vecs = yael.fvec_to_numpy(feature_vecs, int(actual_nuse) * dimension)
        elif file_format == 'bvecs':
            part_np_feature_vecs = yael.bvec_to_numpy(feature_vecs, int(actual_nuse) * dimension)

        part_np_feature_vecs = part_np_feature_vecs.reshape((int(actual_nuse), dimension))

        if run_index != 'y':
            if np_feature_vecs != None:
                np_feature_vecs = numpy.concatenate((np_feature_vecs, part_np_feature_vecs))
            else:
                np_feature_vecs = part_np_feature_vecs
        else:
            index(lsh, part_np_feature_vecs, actual_total_nuse)        
            del part_np_feature_vecs
            if index_folder != None:
                save_index(lsh, index_folder, feature_idx_begin)

        actual_total_nuse += int(actual_nuse)

    if run_index != 'y':
        print np_feature_vecs.shape

    return np_feature_vecs


def index(lsh, np_feature_vecs, label_idx):

    print "indexing..."

    for vec in np_feature_vecs:
        lsh.index(vec, extra_data = 'vec' + str(label_idx))
        label_idx += 1

    print "indexing done."

    return label_idx

def save_index(lsh, base_folder, file_index):

    print "saving index file to " + base_folder + '/' + str(file_index)
    
    if not os.access(base_folder, os.R_OK):    
        os.makedirs(base_folder)
    lsh.save_index(base_folder + '/' + str(file_index))

    print "saving done."


def main():

    parser = argparse.ArgumentParser(description = 'Tools for hamming distance-based image retrieval by cuda')
    parser.add_argument('-f', help = 'The filename of image raw features (SIFT).')
    parser.add_argument('-v', default = 'fvecs', help = 'The format of image raw features.')
    parser.add_argument('-s', default = 'dict', help = 'The method of indexing storage.')
    parser.add_argument('-d', default = '128', help = 'Dimensions of raw image feature.')
    parser.add_argument('-o', default = '0', help = 'Offset of accessing raw image features.')
    parser.add_argument('-n', default = '1', help = 'Number of raw image features to read.')
    parser.add_argument('-i', default = 'n', help = 'Whether to perform indexing step.')
    parser.add_argument('-e', help = 'The dirname of indexing folder.')
    parser.add_argument('-k', default = '10', help = 'Number of retrieved images.')

    args = parser.parse_args()

    d = int(args.d)
    nuse = int(args.n)
    off = int(args.o)

    lsh = LSHash(64, d, 1, storage_config = args.s, matrices_filename = 'project_plane.npz')
    np_feature_vecs = load_features(args.f, args.v, nuse, d, lsh, args.e, off, args.i)

    if args.i == 'y':
        #index(lsh, np_feature_vecs, off)
        if args.e != None and (args.s == 'dict' or args.s == 'random'):
            lsh.save_index(args.e)
    else:
        if args.e != None and (args.s == 'dict' or args.s == 'random'):
            lsh.load_index(args.e)
        elif args.s != 'redis':
            print "Please specify generated indexing file, or use redis mode."
            sys.exit(0)

        retrived = lsh.query(np_feature_vecs[0], num_results = int(args.k), distance_func = 'hamming')
        print retrived

if __name__ == "__main__":
    main()

