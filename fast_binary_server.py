
import numpy
import yutils
import yael
import timeit
import time
import argparse

from flask import Flask

from lshash import LSHash


def load_features(filename, file_format, total_nuse, dimension, offset = 0):

    np_feature_vecs = []
    actual_total_nuse = 0

    for feature_idx_begin in range(offset, total_nuse + offset, 10000000):

        print "loading from " + str(feature_idx_begin)

        nuse = 0
        if (total_nuse + offset) > (feature_idx_begin + 10000000):
            nuse = 10000000
        else:
            nuse = (total_nuse + offset) - feature_idx_begin

        (feature_vecs, actual_nuse) = yutils.load_vectors_fmt(filename, file_format, dimension, nuse, feature_idx_begin , verbose = True)

        actual_total_nuse += int(actual_nuse)

        part_np_feature_vecs = None

        if file_format == 'fvecs':
            part_np_feature_vecs = yael.fvec_to_numpy(feature_vecs, int(actual_nuse) * dimension)
        elif file_format == 'bvecs':
            part_np_feature_vecs = yael.bvec_to_numpy(feature_vecs, int(actual_nuse) * dimension)

        np_feature_vecs = numpy.concatenate((np_feature_vecs, part_np_feature_vecs))
    
    np_feature_vecs = np_feature_vecs.reshape((actual_total_nuse, dimension))

    print np_feature_vecs.shape

    return np_feature_vecs


def index(lsh, np_feature_vecs, label_idx):

    print "indexing..."

    for vec in np_feature_vecs:
        lsh.index(vec, extra_data = 'vec' + str(label_idx))
        label_idx += 1

    print "indexing done."

    return label_idx


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

np_feature_vecs = load_features(args.f, args.v, nuse, d, off)

lsh = LSHash(64, d, 1, storage_config = 'redis', matrices_filename = 'project_plane.npz')

if args.i == 'y':
    index(lsh, np_feature_vecs, off)

# initiate API server
app = Flask(__name__)
@app.route('/search/<vec_id>', methods = ['GET'])
def search(vec_id):
    if long(vec_id) < np_feature_vecs.shape[0]:
        retrived = lsh.query(np_feature_vecs[long(vec_id)], num_results = int(args.k), distance_func = 'hamming')
    else:
        print "out of query vectors' range."

if __name__ == "__main__":
    app.run(debug = True)

