 
import sys
import numpy
import yutils
import yael
import timeit
import time
import os
import argparse

from lshash import LSHash

def cal_recall(retrived, ground_truth, query_idx, topN = 100, topGT = 10):

    results = []
    for result in retrived[0:topN]:
        results.append(result[0])

    results = numpy.array(results)
    print "shape: ", results.shape

    founds = 0
    max_gt_range = ground_truth[query_idx].shape[0]
    if topGT < max_gt_range: max_gt_range = topGT

    for ground_t_idx in range(0, max_gt_range):
        ground_t = ground_truth[query_idx][ground_t_idx]
        found = numpy.where(results == ground_t)
        if len(found[0]) > 0:
            founds += 1

    print "Found: ", founds
    return founds


def load_ground_truth(filename, file_format, dimension = 1000, nuse = 10000, offset = 0):
 
    (feature_vecs, actual_nuse) = yutils.load_vectors_fmt(filename, file_format, dimension, nuse, offset, verbose = True)
    feature_vecs = yael.ivec_to_numpy(feature_vecs, int(actual_nuse) * dimension)
    feature_vecs = feature_vecs.reshape((int(actual_nuse), dimension))

    return (feature_vecs, actual_nuse)
 

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

        # for CUDA-based batch indexing, skip the reshaping
        #part_np_feature_vecs = part_np_feature_vecs.reshape((int(actual_nuse), dimension))

        if run_index != 'y':
            part_np_feature_vecs = part_np_feature_vecs.reshape((int(actual_nuse), dimension))

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

    # batch indexing by CUDA
    lsh.cuda_index(np_feature_vecs, label_idx)

    print "indexing done."

    return label_idx

def save_index(lsh, base_folder, file_index):

    print "saving index file to " + base_folder + '/' + str(file_index)
    
    if not os.access(base_folder, os.R_OK):    
        os.makedirs(base_folder)
    lsh.save_index(base_folder + '/' + str(file_index))

    print "saving done."

def parse_parameters():
 
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
    parser.add_argument('-r', default = '32', help = 'Number of dimensions randomly sampled.')
    parser.add_argument('-c', default = 'n', help = 'Whether to perform compressing step.')
    parser.add_argument('-q', default = 'n', help = 'Whether to sequentially sampling.')
    parser.add_argument('-p', default = 'n', help = 'Whether to perform querying in compressed domain.')
    parser.add_argument('-g', default = 'y', help = 'GPU mode. default is "yes".')
    parser.add_argument('-l', default = 'n', help = 'VLQ base64 mode. Load VLQ base64 encoding compressed dict.')
    parser.add_argument('-b', default = '1', help = 'Expanding level of search buckets.')
    parser.add_argument('-t', default = 'int32', help = 'FastDict type (int32, int8, string).')
    parser.add_argument('-u', default = 'local', help = 'CUDA client type (local, net).')
    parser.add_argument('-host', default = 'localhost', help = 'CUDA server address.')
    parser.add_argument('-title', default = 'Run experiments', help = 'Experiment title.')
    parser.add_argument('-gt', help = 'Ground Truth file.')
    parser.add_argument('-gtf', default = 'ivecs', help = 'Ground Truth file format.')
 

    args = parser.parse_args()

    return args

def init_lsh(args):
 
    d = int(args.d)
    nuse = int(args.n)
    off = int(args.o)
    random_dims = int(args.r)

    random_sampling = True
    if args.q == 'y':
        random_sampling = False

    lsh = LSHash(64, d, random_sampling, args.t, args.u, args.host, random_dims, 1, storage_config = args.s, matrices_filename = 'project_plane.npz')

    np_feature_vecs = load_features(args.f, args.v, nuse, d, lsh, args.e, off, args.i)
 
    return (lsh, np_feature_vecs)

def run(args, lsh):

    if args.c == 'y':
        if args.e != None and args.s == 'random':
            lsh.load_index(args.e)
            print "compressing index..."
            lsh.compress_index(args.e)
            print "compressing done."
        else:
            print "Please specify generated indexing file."
            sys.exit(0)

    if args.c != 'y' and args.i != 'y' and args.e != None and args.s == 'random':
        if args.p == 'y':
            print "loading compressed index."
            lsh.load_compress_index(args.e, (args.l == 'y'))
            print "loading done."
        else:
            print "loading index."
            lsh.load_index(args.e)
            print "loading done."

    return lsh

            
 



 
