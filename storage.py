# lshash/storage.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

import json
import numpy as np
import struct

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
        self.storage = dict()

        self.init_key_dimension(config['r'], config['dim'])
        self.init_bases(config['r'])

    def init_key_dimension(self, num_of_r, dim):
        self.key_dimensions = np.sort(np.random.choice(dim, num_of_r, replace = False))

    def init_bases(self, num_of_r):
        self.bases = np.left_shift(1, range(0, num_of_r))

    def neighbor_keys(self, key):
        actual_key = self.actual_key(key)
        return np.bitwise_xor(actual_key, self.bases)

    def actual_key(self, key):
        key_binary = np.binary_repr(key, width = 64)
        bits = bitarray(key_binary)

        actual_key_bits = []
        for dim in self.key_dimensions:
            actual_key_bits.append(bits[dim])

        actual_key_binary = bitarray(actual_key_bits)
        string = struct.unpack("<I", actual_key_binary.tobytes())[0]
        actual_key = np.array([string]).astype(np.uint32)[0]

        return actual_key
 
    def set_val(self, key, val):
        actual_key = self.actual_key(key)
        self.storage[actual_key] = (key, val)

    def get_val(self, key):
        actual_key = self.actual_key(key)
        return self.storage[actual_key]

    def append_val(self, key, val):
        actual_key = self.actual_key(key)
        self.storage.setdefault(actual_key, []).append((key, val))

    def get_list(self, key, filter_code):
        actual_key = self.actual_key(key)

        vals = []
        for key_value in self.storage[actual_key]:
            if filter_code == key_value[0]:
                vals.append(key_value[1])

        return vals

    def keys(self, reference_key, snd_extend = True):
        neighbor_keys = self.neighbor_keys(reference_key)

        if snd_extend == True:
            extends_n_keys = neighbor_keys
            for n_key in neighbor_keys:
                extends_n_keys = np.append(extends_n_keys, np.bitwise_xor(n_key, self.bases))
            
            neighbor_keys = extends_n_keys

        actual_key = self.actual_key(reference_key)
        all_keys = np.unique(np.append(neighbor_keys, actual_key))
        
        keys = []
        for short_key in all_keys:
            if short_key in self.storage:
                for key_value in self.storage[short_key]:
                    keys.append(key_value[0])

        return keys
 
    def get_neighbor_vals(self, key):
        neighbor_keys = self.neighbor_keys(key)
        vals = []
        for neighbor_key in neighbor_keys:
            vals.append(self.storage[neighbor_key])

        return np.array(vals)
    
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
