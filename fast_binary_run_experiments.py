
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
        
        b_begin = int(args.b)
        b_end = int(args.b) + 1
        # when given a range of expanding levels
        if args.b_begin != -1 and args.b_end != -1:
            b_begin = int(args.b_begin)
            b_end = int(args.b_end)

        for cur_expand_level in range(b_begin, b_end):

            client.send_query(['reset'])
            client.send_query([args.title])
            client.send_query(['expand_level: ' + str(cur_expand_level)])
            
            total_found = {'10': 0, '100': 0}
            for feature_idx in range(0, np_feature_vecs.shape[0]):
            
                feature = np_feature_vecs[feature_idx]
            
                if args.p != 'y':
                    retrived = lsh.query(feature, num_results = int(args.k), expand_level = cur_expand_level, distance_func = 'hamming')
                else:
                    retrived = lsh.query_in_compressed_domain(feature, num_results = int(args.k), expand_level = cur_expand_level, distance_func = 'hamming', gpu_mode = args.g, vlq_mode = args.l)
            
                total_found['10'] += cal_recall(retrived, ground_truth, feature_idx, int(args.k), topGT = 10)
                total_found['100'] += cal_recall(retrived, ground_truth, feature_idx, int(args.k), topGT = 100)
 
            
            recall_r = total_found['10'] / float(np_feature_vecs.shape[0])
            recall_r_str = "recall@" + args.k + " GT@10: " + str(recall_r)
            print recall_r_str
            client.send_query([recall_r_str])
 
            recall_r = total_found['100'] / float(np_feature_vecs.shape[0])
            recall_r_str = "recall@" + args.k + " GT@100: " + str(recall_r)
            print recall_r_str
            client.send_query([recall_r_str])
 

if __name__ == "__main__":
    main()

