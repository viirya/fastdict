import numpy
import array
import time
import math
import argparse
import conn

import socket

def cudaclient(client_type = 'local', options = {}):

    if client_type == 'local':
        from cuda_hamming import CudaHamming
        return CudaHamming()
    elif client_type == 'net':
        if options == {}:
            options = {'host': 'localhost', 'port': 8080}
        return CudaHammingNetClient()
    else:
        raise ValueError("CUDA Client must be local or net type.")


class CudaHammingNetClient(object):

    def __init__(self, options = {'host': 'localhost', 'port': 8080}):

        self.host = options['host']
        self.port = options['port']

    # two numpy array of dtype uint64
    def multi_iteration(self, vec_a, vec_b):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))

        if s.sendall('multi_iteration') == None:
            data = s.recv(1024)
            if data != 'next': raise ValueError('Socket Error')

            if s.sendall(str(vec_a.shape[0] * 8)) == None:
                data = s.recv(1024)
                if data != 'next': raise ValueError('Socket Error')

 
                if s.sendall(vec_a) == None:
                    data = s.recv(1024)
                    if data != 'next': raise ValueError('Socket Error')
                    conn.send_long_vector(s, vec_b)
                else:
                    raise ValueError('Socket Error') 
            else:
                raise ValueError('Socket Error') 
        else:
            raise ValueError('Socket Error') 
                
        print "ready to receive hamming results." 

        # receive results
        if s.sendall('ready') == None:

            distances = conn.recv_long_vector(s, numpy.uint8)
            print distances
            print distances.shape
            return distances

        else:
            raise ValueError('Socket Error')
            
 
    # vec_a: numpy array of dtype uint64
    # compressed_columns_vec: array of buffers
    def cuda_hamming_dist_in_compressed_domain(self, vec_a, compressed_columns_vec, image_ids, vlq_mode):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))

        if s.sendall('cuda_hamming_dist_in_compressed_domain') == None:
            data = s.recv(1024)
            if data != 'next': raise ValueError('Socket Error')

            if s.sendall(str(vec_a.shape[0] * 8)) == None:
                data = s.recv(1024)
                if data != 'next': raise ValueError('Socket Error')

 
                if s.sendall(vec_a) == None:
                    data = s.recv(1024)
                    if data != 'next': raise ValueError('Socket Error')

                    # tell server how many columns_vector to send
                    if s.sendall(str(len(compressed_columns_vec))) == None:
                        data = s.recv(1024)
                        if data != 'next': raise ValueError('Socket Error')
                    else:
                        raise ValueError('Socket Error') 
                
                    for columns in compressed_columns_vec:   
                        print 'columns leg: ', len(columns)

                        # how many columns in a columns vector
                        # should be 64 columns for 64 bits
                        if s.sendall(str(len(columns))) == None:
                            data = s.recv(1024)
                            if data != 'next': raise ValueError('Socket Error')
                        else:
                            raise ValueError('Socket Error') 
 
                        for column in columns:

                            if vlq_mode == 'y':
                                conn.send_long_vector(s, numpy.frombuffer(column, dtype = numpy.uint8), 1)
                            else:
                                conn.send_long_vector(s, numpy.frombuffer(column, dtype = numpy.uint32), 4)
 
                    if s.sendall("done") == None:
                        data = s.recv(1024)
                        if data != 'ok': raise ValueError('Socket Error')
                    else:
                        raise ValueError('Socket Error')
                else:
                    raise ValueError('Socket Error')
        else:
            raise ValueError('Socket Error')

        # send image id length
        if s.sendall(str(len(image_ids))) == None:
            data = s.recv(1024)
            if data != 'next': raise ValueError('Socket Error')
            if s.sendall(vlq_mode) == None:
                if s.recv(1024) != 'done': raise ValueError('Socket Error')
            else:
                raise ValueError('Socket Error') 
        else:
            raise ValueError('Socket Error')

        # receive results
        if s.sendall('ready') == None:

            distances = conn.recv_long_vector(s, numpy.uint8)
            print distances
            print distances.shape
            return distances

        else:
            raise ValueError('Socket Error')
            
 
    def send_query(self, datas):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))

        for data in datas:
            s.send(data)

        r_data = s.recv(1024)
        s.close()

        print "Received", r_data

if __name__ == "__main__":

    client = cudaclient('net')
    client.send_query([buffer('test')])
    client.cuda_hamming_dist_in_compressed_domain(numpy.array([123]), [[numpy.array([123]), numpy.array([456])]], [], True)

