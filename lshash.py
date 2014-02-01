# lshash/lshash.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

import os
import json
import numpy as np
import struct
import time
import re
import pickle

from storage import storage

from cuda_hamming import CudaHamming
from cuda_indexing import CudaIndexing

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None


class LSHash(object):
    """ LSHash implments locality sensitive hashing using random projection for
    input vectors of dimension `input_dim`.

    Attributes:

    :param hash_size:
        The length of the resulting binary hash in integer. E.g., 32 means the
        resulting binary hash will be 32-bit long.
    :param input_dim:
        The dimension of the input vector. E.g., a grey-scale picture of 30x30
        pixels will have an input dimension of 900.
    :param num_hashtables:
        (optional) The number of hash tables used for multiple lookups.
    :param storage_config:
        (optional) A dictionary of the form `{backend_name: config}` where
        `backend_name` is the either `dict` or `redis`, and `config` is the
        configuration used by the backend. For `redis` it should be in the
        format of `{"redis": {"host": hostname, "port": port_num}}`, where
        `hostname` is normally `localhost` and `port` is normally 6379.
    :param matrices_filename:
        (optional) Specify the path to the compressed numpy file ending with
        extension `.npz`, where the uniform random planes are stored, or to be
        stored if the file does not exist yet.
    :param overwrite:
        (optional) Whether to overwrite the matrices file if it already exist
    """

    def __init__(self, hash_size, input_dim, random_sampling = True, dict_type = 'int32', random_dims = 32, num_hashtables=1,
                 storage_config=None, matrices_filename=None, overwrite=False):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables

        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = {storage_config: {}}

        if storage_config == 'random':
            self.storage_config = {'random': {'r': random_dims, 'dim': hash_size, 'random': random_sampling, 't': dict_type}}

        if matrices_filename and not matrices_filename.endswith('.npz'):
            raise ValueError("The specified file name must end with .npz")
        self.matrices_filename = matrices_filename
        self.overwrite = overwrite

        self._init_uniform_planes()
        self._init_hashtables()

        self.loaded_keys = None

        self.cuda_hamming = CudaHamming()

        self.cuda_indexing = CudaIndexing()

    def _init_uniform_planes(self):
        """ Initialize uniform planes used to calculate the hashes

        if file `self.matrices_filename` exist and `self.overwrite` is
        selected, save the uniform planes to the specified file.

        if file `self.matrices_filename` exist and `self.overwrite` is not
        selected, load the matrix with `np.load`.

        if file `self.matrices_filename` does not exist and regardless of
        `self.overwrite`, only set `self.uniform_planes`.
        """

        if "uniform_planes" in self.__dict__:
            return

        if self.matrices_filename:
            file_exist = os.path.isfile(self.matrices_filename)
            if file_exist and not self.overwrite:
                try:
                    npzfiles = np.load(self.matrices_filename)
                except IOError:
                    print("Cannot load specified file as a numpy array")
                    raise
                else:
                    npzfiles = sorted(npzfiles.items(), key=lambda x: x[0])
                    self.uniform_planes = [t[1] for t in npzfiles]
            else:
                self.uniform_planes = [self._generate_uniform_planes()
                                       for _ in xrange(self.num_hashtables)]
                try:
                    np.savez_compressed(self.matrices_filename,
                                        *self.uniform_planes)
                except IOError:
                    print("IOError when saving matrices to specificed path")
                    raise
        else:
            self.uniform_planes = [self._generate_uniform_planes()
                                   for _ in xrange(self.num_hashtables)]

    def _init_hashtables(self):
        """ Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """

        self.hash_tables = [storage(self.storage_config, i)
                            for i in xrange(self.num_hashtables)]

    def _generate_uniform_planes(self):
        """ Generate uniformly distributed hyperplanes and return it as a 2D
        numpy array.
        """

        return np.random.randn(self.hash_size, self.input_dim)

    def _hash(self, planes, input_point):
        """ Generates the binary hash for `input_point` and returns it.

        :param planes:
            The planes are random uniform planes with a dimension of
            `hash_size` * `input_dim`.
        :param input_point:
            A Python tuple or list object that contains only numbers.
            The dimension needs to be 1 * `input_dim`.
        """

        try:
            input_point = np.array(input_point)  # for faster dot product
            projections = np.dot(planes, input_point)
        except TypeError as e:
            print("""The input point needs to be an array-like object with
                  numbers only elements""")
            raise
        except ValueError as e:
            print("""The input point needs to be of the same dimension as
                  `input_dim` when initializing this LSHash instance""", e)
            raise
        else:
            string = "".join(['1' if i > 0 else '0' for i in projections])
            string = struct.unpack(">Q", bitarray(string).tobytes())[0]
            binary_hash = np.array([string]).astype(np.uint64)
            return binary_hash[0] # bitarray(string).tobytes()

    def _as_np_array(self, json_or_tuple):
        """ Takes either a JSON-serialized data structure or a tuple that has
        the original input points stored, and returns the original input point
        in numpy array format.
        """
        if isinstance(json_or_tuple, basestring):
            # JSON-serialized in the case of Redis
            try:
                # Return the point stored as list, without the extra data
                tuples = json.loads(json_or_tuple)[0]
            except TypeError:
                print("The value stored is not JSON-serilizable")
                raise
        else:
            # If extra_data exists, `tuples` is the entire
            # (point:tuple, extra_data). Otherwise (i.e., extra_data=None),
            # return the point stored as a tuple
            tuples = json_or_tuple

        if isinstance(tuples[0], tuple):
            # in this case extra data exists
            return np.asarray(tuples[0])

        elif isinstance(tuples, (tuple, list)):
            try:
                return np.asarray(tuples)
            except ValueError as e:
                print("The input needs to be an array-like object", e)
                raise
        else:
            raise TypeError("query data is not supported")

    def index(self, input_point, extra_data=None):
        """ Index a single input point by adding it to the selected storage.

        If `extra_data` is provided, it will become the value of the dictionary
        {input_point: extra_data}, which in turn will become the value of the
        hash table. `extra_data` needs to be JSON serializable if in-memory
        dict is not used as storage.

        :param input_point:
            A list, or tuple, or numpy ndarray object that contains numbers
            only. The dimension needs to be 1 * `input_dim`.
            This object will be converted to Python tuple and stored in the
            selected storage.
        :param extra_data:
            (optional) Needs to be a JSON-serializable object: list, dicts and
            basic types such as strings and integers.
        """

        if isinstance(input_point, np.ndarray):
            input_point = input_point.tolist()

        #if extra_data != None:
        #    value = (tuple(input_point), extra_data)
        #else:
        #    value = tuple(input_point)

        # customised: we only care about extra_data
        value = (extra_data)

        for i, table in enumerate(self.hash_tables):
            table.append_val(self._hash(self.uniform_planes[i], input_point),
                             value)
    def cuda_index(self, input_points, extra_data = None):

        # we have only one table
        indexed_data = self.cuda_indexing.batch_indexing(self.uniform_planes[0], input_points)

        #for data in indexed_data:
        #    self.hash_tables[0].append_val(data, extra_data)
        #    extra_data += 1

        self.hash_tables[0].batch_append_vals(indexed_data.tolist(), extra_data)
            

    def load_index(self, dirname):

        print "loading index..."

        if 'random' in self.storage_config:
            onlyfiles = [ f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) ]

            for afile in onlyfiles:
                m = re.search('(.*)_(\d)\.dict', afile)

                if m != None:
                
                    print "loading " + dirname + '/' + afile + " ..."

                    self.hash_tables[int(m.group(2))].load(dirname + '/' + afile)

                #for i, table in enumerate(self.hash_tables):
                #    table.load(filename + "_" + str(i) + ".dict")

            print "loading done."

            return

        file_exist = os.path.isfile(filename)
        if file_exist:
            try:
                #npzfiles = np.load(filename)
                f = open(filename)
                self.hash_tables = pickle.load(f)
                self.load_keys()
                f.close()
            except IOError:
                print("Cannot load specified file as a numpy array")
                raise
            #else:
            #    npzfiles = sorted(npzfiles.items(), key=lambda x: x[0])
            #    self.hash_tables = [t[1] for t in npzfiles]

    def compress_index(self, dirname):
        if 'random' in self.storage_config:
            for i, table in enumerate(self.hash_tables):
                table.compress()
                table.save(dirname + '/' + "compressed.cdict")

                table.to_VLQ_base64()
                table.save(dirname + '/' + "compressed_vlq.cdict")
 
                table.clear()

    def load_compress_index(self, dirname, vlq = False):
        if 'random' in self.storage_config:
            for i, table in enumerate(self.hash_tables):
                if not vlq:
                    table.load(dirname + '/' + "compressed.cdict")
                else:
                    print "loading VLQ base64 version..."
                    table.load(dirname + '/' + "compressed_vlq.cdict")

    def save_index(self, filename):

        if 'random' in self.storage_config:
            for i, table in enumerate(self.hash_tables):
                table.save(filename + "_" + str(i) + ".dict")
                table.clear()
            return

        f = open(filename, 'w')

        tables = [table
                  for i, table in enumerate(self.hash_tables)]

        try:
            #np.savez_compressed(filename, tables)
            pickle.dump(tables, f)
            f.close()
        except IOError:
            print("IOError when saving matrices to specificed path")
            raise

    def load_keys(self, key = None, expand_level = 1):

        if 'random' in self.storage_config and key == None: 
            return

        print "loading keys..."
        (keys, image_ids) = self.hash_tables[0].keys(key, expand_level)

        return (np.array(keys).astype(np.uint64), image_ids)


    def fetch_extra_data(self, hamming_candidates):

        table = self.hash_tables[0]
        candidates = []
        for cand in hamming_candidates:
            key = cand[0]
            dist = cand[1]
            candidates.append([key, table.get_list(key, key), dist])

        return candidates


    def query_in_compressed_domain(self, query_point, num_results=None, expand_level = 1, distance_func=None, gpu_mode = 'y', vlq_mode = 'n'):

        if distance_func == "hamming":

            if not bitarray:
                raise ImportError(" Bitarray is required for hamming distance")

            if self.num_hashtables == 1:
                binary_hash = np.array([self._hash(self.uniform_planes[0], query_point)]).astype(np.uint64)

                if 'random' in self.storage_config:

                    if gpu_mode == 'n':
                        print "cpu-based uncompressing..."
                        start = time.clock()

                        b_codes = self.hash_tables[0].uncompress_binary_codes(binary_hash, expand_level)

                        binary_codes = []
                        for binary_code in b_codes.first:
                            #print long(binary_code)
                            binary_codes.append(str(binary_code))

                        elapsed = (time.clock() - start)
                        print "time: " + str(elapsed)

                        binary_codes = np.array(binary_codes).astype(np.uint64)
                        hamming_distances = self.query_with_binary_codes(binary_hash, binary_codes, num_results)

                        return self.sorting(b_codes.second, hamming_distances)

                    else:

                        if vlq_mode == 'n':
                            self.hash_tables[0].init_runtime()
                        else:
                            self.hash_tables[0].init_runtime_vlq_base64()

                        (cols_vector, image_ids) = self.hash_tables[0].get_compressed_cols(binary_hash, expand_level)
                        
                        print "cuda processing..."
                        start = time.clock()
                        
                        hamming_distances = self.cuda_hamming.cuda_hamming_dist_in_compressed_domain(binary_hash, cols_vector, image_ids, vlq_mode)
                        
                        elapsed = (time.clock() - start)
                        print "time: " + str(elapsed)

                        return self.sorting(image_ids, hamming_distances)


    def query(self, query_point, num_results=None, expand_level = 1, distance_func=None):
        """ Takes `query_point` which is either a tuple or a list of numbers,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.

        :param query_point:
            A list, or tuple, or numpy ndarray that only contains numbers.
            The dimension needs to be 1 * `input_dim`.
            Used by :meth:`._hash`.
        :param num_results:
            (optional) Integer, specifies the max amount of results to be
            returned. If not specified all candidates will be returned as a
            list in ranked order.
        :param distance_func:
            (optional) The distance function to be used. Currently it needs to
            be one of ("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
            will used.
        """

        if distance_func == "hamming" and 'random' in self.storage_config:
            if not bitarray:
                raise ImportError(" Bitarray is required for hamming distance")

            if self.num_hashtables == 1:
                d_func = LSHash.hamming_dist
                binary_hash = np.array([self._hash(self.uniform_planes[0], query_point)]).astype(np.uint64)

                print "fetch keys..."
                start = time.clock()
                (binary_codes, image_ids) = self.load_keys(binary_hash, expand_level)
                elapsed = (time.clock() - start)
                print "time: " + str(elapsed)

                print binary_codes.shape

                hamming_distances = self.query_with_binary_codes(binary_hash, binary_codes, num_results)

                return self.sorting(image_ids, hamming_distances)


    def sorting(self, hamming_candidates, hamming_distances, num_results = None):
       
        self.benchmark_begin("sorting")
 
        hamming_results = []
        for idx in range(0, len(hamming_distances)):
            hamming_results.append((hamming_candidates[idx], hamming_distances[idx]))

        hamming_results.sort(key=lambda x: x[1]) 

        self.benchmark_end("sorting")

        hamming_results = hamming_results[:num_results] if isinstance(num_results, int) else hamming_results
        
        return hamming_results
        

    def query_with_binary_codes(self, binary_hash, binary_codes, num_results):

        print "cuda processing..."
        start = time.clock()
        
        hamming_distances = self.cuda_hamming.multi_iteration(binary_hash, binary_codes)
        
        elapsed = (time.clock() - start)
        print "time: " + str(elapsed)

        return hamming_distances
        

    def benchmark_begin(self, title):
        print "start to " + title
        self.start = time.clock()

    def benchmark_end(self, title):
        print "end of " + title
        elapsed = (time.clock() - self.start)
        print "time: " + str(elapsed)

    ### distance functions

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
