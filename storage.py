# lshash/storage.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

import json
import numpy as np
import time
import struct

import fastdict

from bitarray import bitarray

try:
    import redis
except ImportError:
    redis = None

__all__ = ['storage']


def storage(storage_config, index):
    """ Given the configuration for storage and the index, return the
    configured storage instance.
    """
    if 'dict' in storage_config:
        return InMemoryStorage(storage_config['dict'])
    elif 'random' in storage_config:
        return RandomInMemoryStorage(storage_config['random'])
    elif 'redis' in storage_config:
        storage_config['redis']['db'] = index
        return RedisStorage(storage_config['redis'])
    else:
        raise ValueError("Only in-memory dictionary and Redis are supported.")


class BaseStorage(object):
    def __init__(self, config):
        """ An abstract class used as an adapter for storages. """
        raise NotImplementedError

    def keys(self):
        """ Returns a list of binary hashes that are used as dict keys. """
        raise NotImplementedError

    def set_val(self, key, val):
        """ Set `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def get_val(self, key):
        """ Return `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def append_val(self, key, val):
        """ Append `val` to the list stored at `key`.

        If the key is not yet present in storage, create a list with `val` at
        `key`.
        """
        raise NotImplementedError

    def get_list(self, key):
        """ Returns a list stored in storage at `key`.

        This method should return a list of values stored at `key`. `[]` should
        be returned if the list is empty or if `key` is not present in storage.
        """
        raise NotImplementedError


class InMemoryStorage(BaseStorage):
    def __init__(self, config):
        self.name = 'dict'
        self.storage = dict()

    def keys(self):
        return self.storage.keys()

    def items(self):
        return self.storage.items()

    def set_val(self, key, val):
        self.storage[key] = val

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val):
        self.storage.setdefault(key, []).append(val)

    def get_list(self, key):
        return self.storage.get(key, [])

class RandomInMemoryStorage(InMemoryStorage):
    def __init__(self, config):
        self.name = 'random'
        self.storage = fastdict.FastCompressUInt32IntDict(config['r'])

        self.load_dict = fastdict.FastCompressUInt32IntDict(config['r'])

        self.init_key_dimension(config['r'], config['dim'], config['random'])
        self.init_bases(config['r'])

        self.config = config

        self.inited_runtime = False
        self.inited_runtime_VLQ_base64 = False

    def init_key_dimension(self, num_of_r, dim, random = True):
        if random:
            self.key_dimensions = np.sort(np.random.choice(dim, num_of_r, replace = False))
        else:
            self.key_dimensions = np.sort(range(0, num_of_r))
        print "key dimensions:"
        print self.key_dimensions
        self.storage.set_keydimensions(self.key_dimensions.tolist())

    def init_bases(self, num_of_r):
        self.bases = np.left_shift(1, range(0, num_of_r))

    def neighbor_keys(self, key):
        actual_key = self.actual_key(key)
        return np.bitwise_xor(actual_key, self.bases)

    def actual_key(self, key):
        key_binary = np.binary_repr(key, width = 64)
        bits = bitarray(key_binary)

        actual_key_bits = []
        for empty_dim in range(0, 32 - self.config['r']):
            actual_key_bits.append(False)

        for dim in self.key_dimensions:
            actual_key_bits.append(bits[dim])

        #actual_key_bits = np.zeros(32 - len(actual_key_bits)).astype(np.int).tolist() + actual_key_bits

        actual_key_binary = bitarray(actual_key_bits, endian='big')
        string = struct.unpack(">I", actual_key_binary.tobytes())[0]
        actual_key = np.array([string]).astype(np.uint32)[0]

        return actual_key
 
    def set_val(self, key, val):
        actual_key = self.actual_key(key)
        self.storage.set(int(actual_key), long(key), val)

    def get_val(self, key):
        actual_key = self.actual_key(key)
        return self.storage.get(int(actual_key))

    def benchmark_begin(self, title):
        print "start to " + title
        self.start = time.clock()
 
    def benchmark_end(self, title):
        print "end of " + title
        elapsed = (time.clock() - self.start)
        print "time: " + str(elapsed)

    def append_val(self, key, val):
        actual_key = self.actual_key(key)
        #print "actual_key: " + str(actual_key)
        #print "key: " + str(key)
        #print "val: " + str(val)
        self.storage.append(int(actual_key), long(key), int(val))

    def batch_append_vals(self, keys, val):
        vals = []
        actual_keys = []
        for key in keys:
            actual_keys.append(int(self.actual_key(key)))
            vals.append(val)
            val += 1

        self.storage.batch_append(actual_keys, keys, vals)    


    def get_list(self, key, filter_code):
        actual_key = self.actual_key(key)

        vals = []
        if self.storage.exist(int(actual_key)):
            for key_value in self.storage.get(int(actual_key)):
                if str(filter_code) == str(key_value.first):
                    vals.append(key_value.second)

        return vals

    def expand_key(self, actual_key, level = 1):
        expanded_keys = np.bitwise_xor(actual_key, self.bases)
        if level > 1:
            neighbor_keys = expanded_keys
            for neighbor_key in neighbor_keys:
                expanded_keys = np.append(expanded_keys, self.expand_key(neighbor_key, level - 1))

        return np.unique(np.array(expanded_keys))

    # given sub-sampled key, return all expanded sub-sampled keys
    def actual_keys(self, reference_key, level = 1):
 
        actual_key = self.actual_key(reference_key)
        if level > 0:
            neighbor_keys = self.expand_key(actual_key, level)
            all_keys = np.unique(np.append(neighbor_keys, actual_key)).astype(np.uint32)
        else:
            all_keys = np.array([actual_key]).astype(np.uint32)
        return all_keys
 
    # given sub-sampled key, retrieve all binary codes in corresponding buckets
    def keys(self, reference_key, level = 1):

        all_keys = self.actual_keys(reference_key, level)  

        keys = []

        for key_value in self.storage.mget(all_keys.tolist()):
            keys.append(str(key_value.first))

        return keys
 
    def get_neighbor_vals(self, key):
        neighbor_keys = self.neighbor_keys(key)
        vals = []
        for neighbor_key in neighbor_keys:
            vals.append(self.storage.get(int(neighbor_key)))

        return np.array(vals)

    def init_runtime(self):
        if not self.inited_runtime:
            print "init rumtime dict..."
            if self.storage.get_dict_status() == 0:
                self.storage.init_runtime_dict()
            else:
                print "Incorrect dict mode."
            print "done."
            self.inited_runtime = True

    def init_runtime_vlq_base64(self):
        if not self.inited_runtime_VLQ_base64:
            print "init rumtine VLQ base64 dict..." 
            if self.storage.get_dict_status() == 1:
                self.storage.init_runtime_VLQ_base64_dict()
            else:
                print "Incorrect dict mode."
            print "done."
            self.inited_runtime_VLQ_base64 = True

    def save(self, filename):
        fastdict.save_compress_uint32_int(filename, self.storage)

    def load(self, filename):
        if self.storage.size() > 0:
            fastdict.load_compress_uint32_int(filename, self.load_dict)
            self.storage.merge(self.load_dict) 
            self.load_dict.clear()
        else:
            fastdict.load_compress_uint32_int(filename, self.storage)
            key_dimensions = []
            self.storage.get_keydimensions(key_dimensions)
            self.key_dimensions = np.array(key_dimensions)

    def compress(self):
        if self.storage.get_dict_status() == -1:
            self.storage.go_index()
        else:
            print "Incorrect dict mode."

    def to_VLQ_base64(self):
        if self.storage.get_dict_status() == 0:
            self.storage.to_VLQ_base64_dict()
        else:
            print "Incorrect dict mode."

    def uncompress_binary_codes(self, reference_key, level):
 
        binary_codes = None
        self.benchmark_begin('uncompressing binary codes')
        if self.storage.get_dict_status() == 0:
            print "non VLQ base64"
            binary_codes = self.storage.mget_binary_codes(self.actual_keys(reference_key, level).tolist())
        elif self.storage.get_dict_status() == 1:
            print "VLQ base64"
            binary_codes = self.storage.mget_VLQ_base64_binary_codes(self.actual_keys(reference_key, level).tolist())
        else:
            print "Incorrect dict mode."
        self.benchmark_end('uncompressing binary codes') 

        return binary_codes

    def show_uncompressed_keys(self, cols_buffer):
        index = 0
        for buffers in cols_buffer:
            print index
            for i in range(0, len(buffers) / 8):
                data = ''
                for j in range(i * 8, i * 8 + 8):
                    data = data + buffers[j]
                print data
                print struct.unpack('Q', data)
            index += 1

    # obtain compressed columns for binary codes to be uncompress with GPU
    def get_compressed_cols(self, reference_key, level = 0):
    
        #neighbor_keys = self.neighbor_keys(reference_key)
        #actual_key = self.actual_key(reference_key)
        #all_keys = np.unique(np.append(neighbor_keys, actual_key))

        self.benchmark_begin('load cols')
        #cols = self.storage.get_cols(int(actual_key))

        cols = None
        image_ids = None

        if self.storage.get_dict_status() == 2:
            print "compressed runtime dict"
            cols = self.storage.mget_cols_as_buffer(self.actual_keys(reference_key, level).tolist())
            image_ids = self.storage.mget_image_ids(self.actual_keys(reference_key, level).tolist())
        elif self.storage.get_dict_status() == 3:
            print "VLQ base64 compressed runtime dict"
            cols = self.storage.mget_VLQ_base64_cols_as_buffer(self.actual_keys(reference_key, level).tolist())
            image_ids = self.storage.mget_VLQ_base64_image_ids(self.actual_keys(reference_key, level).tolist())

        self.benchmark_end('load cols')

        #columns = [0] * len(cols.first)
        
        #self.benchmark_begin('cols to np array')

        #for column in cols.first:
        #for index in range(0, len(cols.first)):
        #    columns[index] = np.array(cols.first[index]).astype(np.uint64)

        #self.benchmark_end('cols to np array')
 
        #self.benchmark_begin('cols to np array')
 
        #np_columns = np.array(columns)

        #self.benchmark_end('cols to np array')

        return (cols, image_ids)

    def clear(self):
        self.storage.clear()

    
class RedisStorage(BaseStorage):
    def __init__(self, config):
        if not redis:
            raise ImportError("redis-py is required to use Redis as storage.")
        self.name = 'redis'
        self.storage = redis.StrictRedis(**config)

    def keys(self, pattern="*"):
        return self.storage.keys(pattern)

    def set_val(self, key, val):
        self.storage.set(key, val)

    def get_val(self, key):
        return self.storage.get(key)

    def append_val(self, key, val):
        self.storage.rpush(key, json.dumps(val))

    def get_list(self, key):
        return self.storage.lrange(key, 0, -1)
