#!/usr/bin/env python

import fastdict
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = 'Tools for investigating dict file.')
parser.add_argument('-f', help = 'The filename of dict.')
args = parser.parse_args()

f_dict = fastdict.FastCompressIntDict(32)
fastdict.load_compress_int(args.f, f_dict)
print f_dict.size()
 
for key in f_dict.keys():
    print "key: " + str(key)

    for ele in f_dict.get(key):
        print ele
        print ele.first
        print ele.second

