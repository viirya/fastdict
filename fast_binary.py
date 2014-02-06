
import sys
import numpy
import yutils
import yael
import timeit
import time
import os
import argparse

from common import load_features, index, save_index, parse_parameters, init_lsh, run

from lshash import LSHash

def main():

    args = parse_parameters()
    (lsh, np_feature_vecs) = init_lsh(args)
    lsh = run(args, lsh)

    if args.c != 'y' and args.i != 'y' and args.e != None and args.s == 'random':
        if args.p != 'y':
            retrived = lsh.query(np_feature_vecs[1], num_results = int(args.k), expand_level = int(args.b), distance_func = 'hamming')
        else:
            retrived = lsh.query_in_compressed_domain(np_feature_vecs[1], num_results = int(args.k), expand_level = int(args.b), distance_func = 'hamming', gpu_mode = args.g, vlq_mode = args.l)
        print retrived

if __name__ == "__main__":
    main()

