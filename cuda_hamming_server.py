import numpy
import array
import time
import math
import argparse
import conn

import socket

from cuda_hamming import CudaHamming

cuda_hamming_obj = CudaHamming()

def init():
    parser = argparse.ArgumentParser(description = 'The CUDA Hamming Distance Server')
    parser.add_argument('-b', default = 'localhost', help = 'Binding host')
    parser.add_argument('-p', default = '8080', help = 'Port to listen')
    
    args = parser.parse_args()
    args.p = int(args.p)

    return args
 
def loop(args):
 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((args.b, args.p))
    s.listen(1)

    print "Socket inited."

    while 1:
        (client, address) = s.accept()
        print "Connected by: ", address

        data = client.recv(1024)
        if not data: break

        if data == 'cuda_hamming_dist_in_compressed_domain':
            print "call cuda_hamming_dist_in_compressed_domain"
            call_cuda_hamming_dist_in_compressed_domain(client)
        elif data == 'multi_iteration':
            print "call multi_iteration"
            call_multi_iteration(client)
        else:
            print data


        client.close()

def call_multi_iteration(client):
 
    vec_a = None # the query
    binary_codes = buffer('') # binary codes to match
 
    if client.sendall('next') == None:
        length = client.recv(1024)
        if not length: raise ValueError('Socket Error')

        #print 'length: ', int(length)

        if client.sendall('next') == None:

            # receive query

            vec_a = client.recv(int(length))

            if not vec_a: raise ValueError('Socket Error')

            #print len(vec_a)
            vec_a = numpy.frombuffer(vec_a, dtype = numpy.dtype(numpy.uint64))
            
            if client.sendall('next') == None:

                binary_codes = conn.recv_long_vector(client)
                    
            else:
                raise ValueError('Socket Error')

    else:
        raise ValueError('Socket Error')

    #print vec_a.shape
    #print binary_codes.shape

    data = client.recv(1024)
    if data != 'ready':
        #print data
        raise ValueError('Socket Error')

    # hamming_distances: uint8 numpy array
    hamming_distances = cuda_hamming_obj.multi_iteration(vec_a, binary_codes)
    hamming_distances = hamming_distances.astype(numpy.uint8)

    #print "results:"
    #print hamming_distances
    #print hamming_distances.shape

    conn.send_long_vector(client, hamming_distances, 1)
    
 
def call_cuda_hamming_dist_in_compressed_domain(client):

    vec_a = None
    columns_vector = []

    if client.sendall('next') == None:
        length = client.recv(1024)
        if not length: raise ValueError('Socket Error')

        #print 'length: ', int(length)

        if client.sendall('next') == None:

            # receive query

            vec_a = client.recv(int(length))

            if not vec_a: raise ValueError('Socket Error')

            #print len(vec_a)
            vec_a = numpy.frombuffer(vec_a, dtype = numpy.dtype(numpy.uint64))
            #print vec_a
            
            if client.sendall('next') == None:
                # begin to receive columns
 
                cols_vec_length = client.recv(1024)
                if not cols_vec_length: raise ValueError('Socket Error')
                #print "cols_vec_length: ", cols_vec_length

                cols_vec_length = int(cols_vec_length)

                if client.sendall('next') == None:

                    while cols_vec_length > 0:
                    
                        cols_length = client.recv(1024)
                        if not cols_length: raise ValueError('Socket Error')
                        #print "cols_length: ", cols_length
                    
                        cols_length = int(cols_length)

                        if client.sendall('next') == None:
                    
                            columns = []
                            while cols_length > 0:
                                columns.append(buffer(conn.recv_long_vector(client, None)))
                                cols_length -= 1
                            
                            columns_vector.append(columns)
                            
                            cols_vec_length -= 1
                        else:
                            raise ValueError('Socket Error')
                    
                    done = client.recv(1024)
                    if done != 'done': raise ValueError('Socket Error')
                    if client.sendall('ok') != None:
                        raise ValueError('Socket Error')

                else:
                    raise ValueError('Socket Error')

            else:
                raise ValueError('Socket Error')

        else:
            raise ValueError('Socket Error')

    else:
        raise ValueError('Socket Error')


    image_ids_leng = client.recv(1024)
    if not image_ids_leng: raise ValueError('Socket Error')

    #print "image ids length: ", image_ids_leng

    image_ids = numpy.zeros(int(image_ids_leng)).tolist()

    if client.sendall('next') == None:
        vlq_mode = client.recv(1024)
        if not vlq_mode: raise ValueError('Socket Error')

        if not client.sendall('done') == None:
            raise ValueError('Socket Error')
    else:
        raise ValueError('Socket Error')    

    if client.recv(1024) != 'ready':
        raise ValueError('Socket Error')

    #print vec_a
    #print columns_vector
    #print vlq_mode

    # hamming_distances: uint8 numpy array
    hamming_distances = cuda_hamming_obj.cuda_hamming_dist_in_compressed_domain(vec_a, columns_vector, image_ids, vlq_mode)

    #print "results:"
    #print hamming_distances
    #print hamming_distances.shape

    hamming_distances = hamming_distances.astype(numpy.uint8)

    conn.send_long_vector(client, hamming_distances, 1)


if __name__ == "__main__":
    args = init()
    loop(args)

 
    
