import numpy
import array
import time
import math
import argparse
import conn
import sys

import socket

from cuda_hamming import CudaHamming

cuda_hamming_obj = CudaHamming()

logging_info = {}

s = None

def init():
    parser = argparse.ArgumentParser(description = 'The CUDA Hamming Distance Server')
    parser.add_argument('-b', default = 'localhost', help = 'Binding host')
    parser.add_argument('-p', default = '8080', help = 'Port to listen')
    
    args = parser.parse_args()
    args.p = int(args.p)

    return args

def log_info(hamming_distances):

    cuda_time = hamming_distances[1]

    log(hamming_distances[0].shape[0])
    log('time: ' + str(cuda_time))
    log('max: ' + str(numpy.amax(hamming_distances[0])))
    log('min: ' + str(numpy.amin(hamming_distances[0])))
    log('mean: ' + str(numpy.mean(hamming_distances[0])))

    nears = numpy.where(hamming_distances[0] <= 10)
    nears = len(nears[0])
    log('< 10 nears: ' + str(nears))

    if not 'cuda_time' in logging_info:
        logging_info['cuda_time'] = cuda_time
        logging_info['cuda_run'] = 1
        logging_info['total_nears'] = nears
    else:
        logging_info['cuda_time'] += cuda_time
        logging_info['cuda_run'] += 1
        logging_info['total_nears'] += nears
        
    logging_info['avg nears'] = logging_info['total_nears'] / logging_info['cuda_run']
    logging_info['avg cuda_time'] = logging_info['cuda_time'] / logging_info['cuda_run']

 
def log_logging_info():

    for key in logging_info:
        log(key + ': ' + str(logging_info[key]))
 
def log(msg):

    try:
        sys.stderr.write(str(msg) + "\n")
    except:
        1
 
def loop(args):
 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.b, args.p))
    s.listen(1)

    print "Socket inited."
 

    try: 
        while 1:
            (client, address) = s.accept()
            print "Connected by: ", address
        
            data = client.recv(1024)
            if not data:
                client.close()
                continue
        
            try:
                if data == 'cuda_hamming_dist_in_compressed_domain':
                    print "call cuda_hamming_dist_in_compressed_domain"
                    call_cuda_hamming_dist_in_compressed_domain(client)
                    log_logging_info()
                elif data == 'multi_iteration':
                    print "call multi_iteration"
                    call_multi_iteration(client)
                    log_logging_info()
                else:
                    log(data)
                    print data
                    
            except:
                print "Exception found. Close connection."
        
            client.close()

    except (KeyboardInterrupt, SystemExit):
        for key in logging_info:
            log(key + ': ' + str(logging_info[key]))

        if s != None:
            print "Closing socket."
            s.shutdown(socket.SHUT_RDWR)
            s.close()

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


    data = client.recv(1024)
    if data != 'ready':
        #print data
        raise ValueError('Socket Error')

    # hamming_distances: uint8 numpy array
    hamming_distances = cuda_hamming_obj.multi_iteration(vec_a, binary_codes)

    log_info(hamming_distances)

    hamming_distances = hamming_distances[0].astype(numpy.uint8)

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

    # hamming_distances: uint8 numpy array
    hamming_distances = cuda_hamming_obj.cuda_hamming_dist_in_compressed_domain(vec_a, columns_vector, image_ids, vlq_mode)

    log_info(hamming_distances)
 
    hamming_distances = hamming_distances[0].astype(numpy.uint8)

    conn.send_long_vector(client, hamming_distances, 1)


if __name__ == "__main__":
    args = init()
    loop(args)

