
import sys
import numpy
import yutils
import yael
import timeit
import time
import os
import argparse

import tornado.ioloop
import tornado.web


from lshash import LSHash

def load_features(filename, file_format, total_nuse, dimension, offset = 0):

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

        if np_feature_vecs != None:
            np_feature_vecs = numpy.concatenate((np_feature_vecs, part_np_feature_vecs))
        else:
            np_feature_vecs = part_np_feature_vecs

        actual_total_nuse += int(actual_nuse)

    print np_feature_vecs.shape

    return np_feature_vecs

def init():

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
 
    args = parser.parse_args()
 
    d = int(args.d)
    nuse = int(args.n)
    off = int(args.o)
    random_dims = int(args.r)
 
    random_sampling = True
    if args.q == 'y':
        random_sampling = False

    lsh = LSHash(64, d, random_sampling, args.t, random_dims, 1, storage_config = args.s, matrices_filename = 'project_plane.npz')
    np_feature_vecs = load_features(args.f, args.v, nuse, d, off)

    if args.c != 'y' and args.i != 'y' and args.e != None and args.s == 'random':
        if args.p == 'y':
            print "loading compressed index."
            lsh.load_compress_index(args.e, (args.l == 'y'))
            print "loading done."
        else:
            print "loading index."
            lsh.load_index(args.e)
            print "loading done."

    print "indexing done. Ready for querying."

    return (lsh, np_feature_vecs, args)


(lsh, np_feature_vecs, args) = init()

class QueryHandler(tornado.web.RequestHandler):
    def get(self, image_id):
        self.write("You requested the image: " + image_id)

        if args.p != 'y':
            retrived = lsh.query(np_feature_vecs[long(image_id)], num_results = int(args.k), expand_level = int(args.b), distance_func = 'hamming')
        else:            
            retrived = lsh.query_in_compressed_domain(np_feature_vecs[long(image_id)], num_results = int(args.k), expand_level = int(args.b), distance_func = 'hamming', gpu_mode = args.g, vlq_mode = args.l)

        self.write(str(retrived))
        print retrived
 

application = tornado.web.Application([
    (r"/query/([0-9]+)", QueryHandler),
])    


if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()


