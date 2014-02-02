import numpy
import array
import time
import math
import argparse

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
        else:
            print data


        client.close()

def call_cuda_hamming_dist_in_compressed_domain(client):

    vec_a = None
    columns_vector = []

    if client.sendall('next') == None:
        length = client.recv(1024)
        if not length: raise ValueError('Socket Error')

        print 'length: ', int(length)

        if client.sendall('next') == None:

            # receive query

            vec_a = client.recv(int(length))

            if not vec_a: raise ValueError('Socket Error')

            print len(vec_a)
            vec_a = numpy.frombuffer(vec_a, dtype = numpy.dtype(numpy.uint64))
            print vec_a
            
            if client.sendall('next') == None:
                # begin to receive columns
 
                cols_vec_length = client.recv(1024)
                if not cols_vec_length: raise ValueError('Socket Error')
                print "cols_vec_length: ", cols_vec_length

                cols_vec_length = int(cols_vec_length)

                if client.sendall('next') == None:

                    while cols_vec_length > 0:
                    
                        cols_length = client.recv(1024)
                        if not cols_length: raise ValueError('Socket Error')
                        print "cols_length: ", cols_length
                    
                        cols_length = int(cols_length)

                        if client.sendall('next') == None:
                    
                            columns = []
                            while cols_length > 0:
                                length = client.recv(1024)
                                if not length: raise ValueError('Socket Error')
                                
                                print "length: ", length
                            
                                if client.sendall('next') == None:
                                
                                    column = client.recv(int(length))
                                    if not column: raise ValueError('Socket Error')
                                    columns.append(buffer(column))
                                    
                                    if client.sendall('next') != None:
                                        raise ValueError('Socket Error')
                                    
                                    cols_length -= 1
                                else:
                                    raise ValueError('Socket Error')
                            
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

    print "image ids length: ", image_ids_leng

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

    print vec_a
    print columns_vector
    print vlq_mode

    # hamming_distances: uint8 numpy array
    hamming_distances = cuda_hamming_obj.cuda_hamming_dist_in_compressed_domain(vec_a, columns_vector, image_ids, vlq_mode)

    print "results:"
    print hamming_distances

    if client.sendall(str(hamming_distances.shape[0])) == None:
        if client.recv(1024) == 'next':
            if client.sendall(hamming_distances) == None:    
                return 
            else:
                raise ValueError('Socket Error')
        else:
            raise ValueError('Socket Error')
    else:
        raise ValueError('Socket Error')


if __name__ == "__main__":
    args = init()
    loop(args)

 
    
