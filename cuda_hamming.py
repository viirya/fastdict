import pycuda.autoinit
import pycuda.driver as drv
import numpy
import array
import time
import math

from pycuda.compiler import SourceModule

class CudaHamming(object):

    def __init__(self,  block = (256, 1, 1), grid = (15, 1)):

        vector_len = 100000
 
        self.vlq_compressed_mod = SourceModule("""
typedef unsigned int uint8_t;
typedef unsigned long int uint32_t;
typedef unsigned long long int uint64_t;

//typedef long int int32_t;

__device__ uint8_t base64_find(char ch) {

    char base64_chars[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                           'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                           'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                           's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                           '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};
 
    for (int i = 0; i < 64; i++)
        if (base64_chars[i] == ch)
            return i;
    return 64;
}

__device__ uint32_t decode_VLQ_base64(char* string, uint32_t* offset) {

    uint8_t VLQ_BASE_SHIFT = 5;
    uint8_t VLQ_BASE = 1 << VLQ_BASE_SHIFT;
    uint8_t VLQ_BASE_MASK = VLQ_BASE - 1;
    uint8_t VLQ_CONTINUATION_BIT = VLQ_BASE;

    uint32_t i = *offset;
    
    uint32_t result = 0;        
    uint8_t shift = 0;
    uint8_t continuation, digit;
    
    do {
        //if (string[i] == '\0') {
        //    return -1;
        //}
        digit = base64_find(string[i++]);
        continuation = digit & VLQ_CONTINUATION_BIT;
        digit &= VLQ_BASE_MASK;
        result = result + (digit << shift);
        shift += VLQ_BASE_SHIFT;
    } while (continuation > 0);

    *offset = i;
    return result;
}

__global__ void vlq_compressed_hamming_dist(uint64_t* query, char** vlq_bit_counts, uint64_t* max_length, char** distances)
{

    const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    
    // how many binary codes we uncompress in a cuda thread
    int batch_size = 50;
    
    uint64_t binary_codes[50] = {};
    const uint64_t i_for_batch = i * batch_size;
    
    if (1 < 2 && i_for_batch < max_length[0]) {
    
        for (uint32_t simple_index = 0; (simple_index < batch_size) && (i_for_batch + simple_index < max_length[0]); simple_index++) {
            binary_codes[simple_index] = 0;
        }
        
        uint64_t binary_code = 0x00;
        
        for (uint64_t column_index = 0; column_index < 64; column_index++) {
        
            uint64_t count_for_bits = 0;
            uint32_t bit_count_offset = 0;
            uint8_t bit_type = 0x00;
        
            // the index for currently uncompressing binary code
            uint64_t current_binary_index = 0;
        
            while (count_for_bits <= max_length[0]) {

                count_for_bits += decode_VLQ_base64(vlq_bit_counts[column_index], &bit_count_offset);
        
                if ((count_for_bits > max_length[0]) || (current_binary_index > batch_size))
                    break;
        
                while ((count_for_bits > (i_for_batch + current_binary_index)) && (current_binary_index < batch_size) && (i_for_batch + current_binary_index < max_length[0])) {
                    if (bit_type == 1) {
                        binary_codes[current_binary_index] = binary_codes[current_binary_index] | ((uint64_t)1 << column_index);
                    }
                    // move to next binary code
                    current_binary_index++;
                } 
                bit_type = bit_type ^ 0x01;
            } 
        
        } 
        
        for (int binary_code_index = 0; (binary_code_index < batch_size) && (i_for_batch + binary_code_index < max_length[0]); binary_code_index++) {
        
            if (binary_codes[binary_code_index] > 0x00 || 1) {
        
                uint64_t xor_r = query[0] ^ binary_codes[binary_code_index];
                
                const uint64_t m1  = 0x5555555555555555; 
                const uint64_t m2  = 0x3333333333333333; 
                const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; 
                const uint64_t m8  = 0x00ff00ff00ff00ff; 
                const uint64_t m16 = 0x0000ffff0000ffff; 
                const uint64_t m32 = 0x00000000ffffffff; 
                const uint64_t hff = 0xffffffffffffffff; 
                const uint64_t h01 = 0x0101010101010101; 
                
                xor_r -= (xor_r >> 1) & m1;    
                xor_r = (xor_r & m2) + ((xor_r >> 2) & m2); 
                xor_r = (xor_r + (xor_r >> 4)) & m4;        
        
                distances[0][i_for_batch + binary_code_index] = (uint8_t)((xor_r * h01) >> 56);
                //distances[i_for_batch + binary_code_index] = tmp_binary_codes[binary_code_index];
                //distances[i_for_batch + binary_code_index] = binary_codes[binary_code_index];
                //distances[i_for_batch + binary_code_index] = i_for_batch + binary_code_index + 1;
            }
        }
    }
}
        """)
        
        self.vlq_compressed_hamming_dist = self.vlq_compressed_mod.get_function("vlq_compressed_hamming_dist")
 

        self.compressed_mod = SourceModule("""
typedef unsigned int uint8_t; // sizeof(uint8_t) is 4
typedef unsigned long int uint32_t; // sizeof(uint32_t) is 8
typedef unsigned long long int uint64_t; // sizeof(uint64_t) is 8
__global__ void compressed_hamming_dist(uint64_t* query, uint8_t** bit_counts, uint64_t* max_length, char** distances)
{
    const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    
    // how many binary codes we uncompress in a cuda thread
    int batch_size = 50;
    
    uint64_t binary_codes[50] = {};
    const uint64_t i_for_batch = i * batch_size;
    
    if (1 < 2 && i_for_batch < max_length[0]) {
    
        for (uint32_t simple_index = 0; (simple_index < batch_size) && (i_for_batch + simple_index < max_length[0]); simple_index++) {
            binary_codes[simple_index] = 0;
        }
        
        uint64_t binary_code = 0x00;
        
        for (uint64_t column_index = 0; column_index < 64; column_index++) {
        
            uint64_t count_for_bits = 0;
            uint64_t bit_count_index = 0;
            uint8_t bit_type = 0x00;
        
            // the index for currently uncompressing binary code
            uint64_t current_binary_index = 0;
        
            while (count_for_bits <= max_length[0]) {
                count_for_bits += (uint64_t)bit_counts[column_index][bit_count_index++];
        
                if ((count_for_bits > max_length[0]) || (current_binary_index > batch_size))
                    break;
        
                while ((count_for_bits > (i_for_batch + current_binary_index)) && (current_binary_index < batch_size) && (i_for_batch + current_binary_index < max_length[0])) {
                    if (bit_type == 1) {
                        binary_codes[current_binary_index] = binary_codes[current_binary_index] | ((uint64_t)1 << column_index);
                    }
                    // move to next binary code
                    current_binary_index++;
                } 
                bit_type = bit_type ^ 0x01;
            } 
        
        } 
        
        for (int binary_code_index = 0; (binary_code_index < batch_size) && (i_for_batch + binary_code_index < max_length[0]); binary_code_index++) {
        
            if (binary_codes[binary_code_index] > 0x00 || 1) {
        
                uint64_t xor_r = query[0] ^ binary_codes[binary_code_index];
                
                const uint64_t m1  = 0x5555555555555555; 
                const uint64_t m2  = 0x3333333333333333; 
                const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; 
                const uint64_t m8  = 0x00ff00ff00ff00ff; 
                const uint64_t m16 = 0x0000ffff0000ffff; 
                const uint64_t m32 = 0x00000000ffffffff; 
                const uint64_t hff = 0xffffffffffffffff; 
                const uint64_t h01 = 0x0101010101010101; 
                
                xor_r -= (xor_r >> 1) & m1;    
                xor_r = (xor_r & m2) + ((xor_r >> 2) & m2); 
                xor_r = (xor_r + (xor_r >> 4)) & m4;        
        
                distances[0][i_for_batch + binary_code_index] = (uint8_t)((xor_r * h01) >> 56);
                //distances[i_for_batch + binary_code_index] = tmp_binary_codes[binary_code_index];
                //distances[i_for_batch + binary_code_index] = binary_codes[binary_code_index];
                //distances[i_for_batch + binary_code_index] = i_for_batch + binary_code_index + 1;
            }
        }
    }
}
        """)
        
        self.compressed_hamming_dist = self.compressed_mod.get_function("compressed_hamming_dist")

        self.mod = SourceModule("""
typedef unsigned long long int uint64_t;
__global__ void hamming_dist(uint64_t *a, uint64_t *b, uint64_t *length)
{
    const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t xor_r;

    // how many binary codes we uncompress in a cuda thread
    int batch_size = 50;
    
    uint64_t binary_codes[50] = {};
    const uint64_t i_for_batch = i * batch_size;

    //if (i < %(length)s) {
    for (int binary_code_index = 0; (binary_code_index < batch_size) && (i_for_batch + binary_code_index < length[0]); binary_code_index++) {

        xor_r = a[0] ^ b[i_for_batch + binary_code_index];
        
        const uint64_t m1  = 0x5555555555555555; 
        const uint64_t m2  = 0x3333333333333333; 
        const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; 
        const uint64_t m8  = 0x00ff00ff00ff00ff; 
        const uint64_t m16 = 0x0000ffff0000ffff; 
        const uint64_t m32 = 0x00000000ffffffff; 
        const uint64_t hff = 0xffffffffffffffff; 
        const uint64_t h01 = 0x0101010101010101; 
        
        xor_r -= (xor_r >> 1) & m1;    
        xor_r = (xor_r & m2) + ((xor_r >> 2) & m2); 
        xor_r = (xor_r + (xor_r >> 4)) & m4;        
        
        b[i_for_batch + binary_code_index] = (xor_r * h01) >> 56;
    }

}
""" % {"length": vector_len})

        self.hamming_dist = self.mod.get_function("hamming_dist")

        self.block = block
        self.grid = grid             

    def benchmark_begin(self, title):
        print "start to " + title
        self.start = time.clock()

    def benchmark_end(self, title):
        print "end of " + title
        elapsed = (time.clock() - self.start)
        print "time: " + str(elapsed)

    def alloc_device_memory_for_cols(self, compressed_columns_vec, vlq_mode, max_length):

        concate_col = None
        if vlq_mode == 'n':
            concate_col = numpy.zeros(max_length * 2 * 4).astype(numpy.uint8)
        else:
            concate_col = numpy.zeros(max_length * 2).astype(numpy.uint8)
 
        gpu_alloc_objs = []
        for col_idx in range(0, 64):

            col_len = 0
            for bucket_idx in range(0, len(compressed_columns_vec)):
                if len(compressed_columns_vec[bucket_idx]) > 0:
                    cur_col_len = len(compressed_columns_vec[bucket_idx][col_idx])
                    numpy.copyto(concate_col[col_len:col_len + cur_col_len], compressed_columns_vec[bucket_idx][col_idx])
                    col_len += cur_col_len

            col_ptr = drv.mem_alloc(col_len)
            drv.memcpy_htod(int(col_ptr), concate_col[0:col_len])

            #col_len = 0
            #for bucket_idx in range(0, len(compressed_columns_vec)):
            #    if len(compressed_columns_vec[bucket_idx]) > 0:
            #        drv.memcpy_htod(int(col_ptr) + col_len, compressed_columns_vec[bucket_idx][col_idx])
            #        col_len += len(compressed_columns_vec[bucket_idx][col_idx])

            gpu_alloc_objs.append(col_ptr)

        return gpu_alloc_objs
            

    def cuda_hamming_dist_in_compressed_domain(self, vec_a, compressed_columns_vec, image_ids, vlq_mode):

        binary_code_length = len(image_ids)

        self.benchmark_begin('preparing')

        addresses = [] 
        gpu_alloc_objs = self.alloc_device_memory_for_cols(compressed_columns_vec, vlq_mode, binary_code_length)
        for address in gpu_alloc_objs:
            addresses.append(int(address))

        np_addresses = numpy.array(addresses).astype(numpy.uint64)

        # We have 64 bit address space on GPU that 1 pointer costs 8 bytes
        # todo: do we have better way to figure the size of pointer in python?
        arrays_gpu = drv.mem_alloc(np_addresses.shape[0] * 8)

        drv.memcpy_htod(arrays_gpu, np_addresses)

        # distances

        distances = numpy.zeros(binary_code_length).astype(numpy.uint8)
        distances_addr = drv.to_device(distances)

        distances_gpu = drv.mem_alloc(1 * 8)
        distances_address = numpy.array([int(distances_addr)]).astype(numpy.uint64)

        drv.memcpy_htod(distances_gpu, distances_address)


        length = numpy.array([binary_code_length]).astype(numpy.uint64)
 
        print "total: " + str(binary_code_length) + " compressed binary codes." 

        self.benchmark_end('preparing')
        self.benchmark_begin('cudaing')

        custom_grid = (int(math.ceil(float(binary_code_length) / (50 * 256))), 1)

        print "custom grid: ", custom_grid

        if vlq_mode == 'n':
            print "non VLQ base64 cuda uncompression and hamming distance calculation"
            self.compressed_hamming_dist(
                drv.In(vec_a), arrays_gpu, drv.In(length), distances_gpu,
                block = self.block, grid = custom_grid)
        else:
            print "VLQ base64 cuda uncompression and hamming distance calculation"
            self.vlq_compressed_hamming_dist(
                drv.In(vec_a), arrays_gpu, drv.In(length), distances_gpu,
                block = self.block, grid = custom_grid)
        
        self.benchmark_end('cudaing')

        drv.memcpy_dtoh(distances, distances_addr)

        print distances
        #count = 0
        #for dis in distances:
        #    print "count: " + str(count) + " " + str(dis) + " image: " + str(image_ids[count]) + "."
        #    count += 1
        print distances.shape

        return distances

    def multi_iteration(self, vec_a, vec_b):

        vector_len = vec_b.shape[0]
 
        sections = range(0, vector_len, 10000000)
        sections = sections[1:]

        sub_vec_bs = numpy.split(vec_b, sections)

        dest = numpy.array([])
        for sub_vec in sub_vec_bs:
            sub_dest = self.cuda_hamming_dist(vec_a, sub_vec)
            dest = numpy.concatenate((dest, sub_dest))

        return dest
            
    def cuda_hamming_dist(self, vec_a, vec_b):

        #dest = numpy.zeros_like(vec_b)
        dest = numpy.array(vec_b)
        length = numpy.array([vec_b.shape[0]]).astype(numpy.uint64)
 
        #for d in dest:
        #    print d

        custom_grid = (int(math.ceil(float(length[0]) / (50 * 256))), 1)
        print "custom grid: ", custom_grid
 
        self.hamming_dist(
                drv.In(vec_a), drv.InOut(dest), drv.In(length),
                block = self.block, grid = custom_grid)
        
        print dest
        #for d in dest:
        #    print d
        print dest.shape
        
        return dest

 
