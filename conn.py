
import numpy

send_block_size = 100

def send_long_vector(s, vector, size_per_item = 8):

    for vector_idx in range(0, vector.shape[0] / send_block_size + 1):
        if s.sendall(str(vector[vector_idx * send_block_size:(vector_idx + 1) * send_block_size].shape[0] * size_per_item)) == None:
            data = s.recv(1024)
            if data != 'next': raise ValueError('Socket Error')
    
            if vector[vector_idx * send_block_size:(vector_idx + 1) * send_block_size].shape[0] == 0: continue

            if s.sendall(vector[vector_idx * send_block_size:(vector_idx + 1) * send_block_size]) == None:
                data = s.recv(1024)
                if data != 'next': raise ValueError('Socket Error')
            else:
                raise ValueError('Socket Error')
        else:
            raise ValueError('Socket Error')
    
    if s.sendall('done') != None: raise ValueError('Socket Error')
    data = s.recv(1024)
    if data != 'ok': raise ValueError('Socket Error')

def recv_long_vector(client, type = numpy.uint64):

    binary_codes = buffer('')

    while 1:
        part_binary_codes_length = client.recv(1024)
        #print "part_binary_codes_length: ", part_binary_codes_length

        if not part_binary_codes_length: raise ValueError('Socket Error')
    
        if part_binary_codes_length == 'done': break
    
        part_binary_codes_length = int(part_binary_codes_length)

        if not client.sendall('next') == None: raise ValueError('Socket Error')

        # if it is empty vector, skip next receiving
        if part_binary_codes_length == 0: continue
    
        binary_codes_part = client.recv(part_binary_codes_length)
        if not binary_codes_part: break
    
        if not client.sendall('next') == None: raise ValueError('Socket Error')
    
        binary_codes = binary_codes + binary_codes_part

    if type != None:    
        binary_codes = numpy.frombuffer(binary_codes, dtype = numpy.dtype(type))
    
    if not client.sendall('ok') == None: raise ValueError('Socket Error')

    return binary_codes


