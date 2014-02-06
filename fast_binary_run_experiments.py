
import sys
import numpy
import yutils
import yael
import timeit
import time
import os
import argparse

from common import load_features, index, save_index, parse_parameters, init_lsh, run, cal_recall, load_ground_truth

from cuda_hamming_client import cudaclient

from lshash import LSHash

def main():    

    args = parse_parameters()    
    (lsh, np_feature_vecs) = init_lsh(args)    
    lsh = run(args, lsh)

    ground_truth = None

    if args.gt != None:
        (ground_truth, ground_truth_num) = load_ground_truth(args.gt, args.gtf)

    if args.c != 'y' and args.i != 'y' and args.e != None and args.s == 'random':        
        client = cudaclient('net', {'host': args.host, 'port': 8080})
        client.send_query([args.title])

        total_found = 0
        for feature_idx in range(0, np_feature_vecs.shape[0]):

            feature = np_feature_vecs[feature_idx]

            if args.p != 'y':
                retrived = lsh.query(feature, num_results = int(args.k), expand_level = int(args.b), distance_func = 'hamming')
            else:
                retrived = lsh.query_in_compressed_domain(feature, num_results = int(args.k), expand_level = int(args.b), distance_func = 'hamming', gpu_mode = args.g, vlq_mode = args.l)
            #print retrived

            total_found += cal_recall(retrived, ground_truth, feature_idx, int(args.k))

        recall_r = total_found / float(np_feature_vecs.shape[0])
        recall_r_str = "recall@" + args.k + ": " + str(recall_r)
        print recall_r_str
        client.send_query([recall_r_str])

if __name__ == "__main__":
    main()

