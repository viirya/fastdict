import pycuda.autoinit
import pycuda.driver as drv
import numpy
import array
import time

from pycuda.compiler import SourceModule

class CudaHamming(object):

    def __init__(self,  block = (256, 1, 1), grid = (10, 1)):

        vector_len = 100000
 
        self.vlq_compressed_mod = SourceModule("""
#define BLOCK_SIZE 256

typedef unsigned int uint8_t;
typedef unsigned long int uint32_t;
typedef unsigned long long int uint64_t;
 
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
 
__global__ void vlq_compressed_hamming_dist(uint64_t* query, char** vlq_bit_counts, uint64_t* max_length, uint64_t* distances)
{
    const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    
    // how many binary codes we uncompress in a cuda thread
    int batch_size = 50;
    
    uint64_t local_distances[50];
    
    const uint64_t i_for_batch = i * batch_size;

    // pre-scan bit count matrix to skip redundant scaning for other threads in same block

    // 64 columns; for each column, record the base index as the beginning for threads in same block to scan
    __shared__ uint32_t bit_counts_scan_bases[BLOCK_SIZE / 64][64];
    __shared__ uint8_t bit_types_for_scan_bases[BLOCK_SIZE / 64][64];
    __shared__ uint64_t accu_bit_counts_for_scan_bases[BLOCK_SIZE / 64][64];


    // shared memory index: 64 threads share common block of shared memory
    uint8_t shared_mem_index = threadIdx.x / 64;

    // column id that this thread is responsible to find beginning base
    uint8_t thread_column_index = threadIdx.x % 64;

    if (thread_column_index < 64) {

        uint32_t smallest_binary_code_index = (gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + 64 * shared_mem_index) * batch_size;

        uint64_t count_for_bits = 0;
        uint32_t bit_count_offset = 0;
        uint8_t bit_type = 0x00;
        while (count_for_bits <= max_length[0]) {
            uint32_t prev_bit_count_offset = bit_count_offset;
            uint32_t cur_bit_count = decode_VLQ_base64(vlq_bit_counts[thread_column_index], &bit_count_offset);

            count_for_bits += cur_bit_count;

            if (count_for_bits > smallest_binary_code_index) {
                bit_counts_scan_bases[shared_mem_index][thread_column_index] = prev_bit_count_offset;
                bit_types_for_scan_bases[shared_mem_index][thread_column_index] = bit_type;
                accu_bit_counts_for_scan_bases[shared_mem_index][thread_column_index] = count_for_bits - cur_bit_count;
                break;
            }

            bit_type = bit_type ^ 0x01;                
        }

    }

    // wait for other threads to find scan bases
    __syncthreads();

    
    if (i_for_batch < max_length[0]) {
    
        for (uint64_t distance_idx = 0; distance_idx < batch_size; distance_idx++) {
            local_distances[distance_idx] = 0;
        }
        
        uint64_t base = 1;
        
        for (uint64_t column_index = 0; column_index < 64; column_index++) {
        
            uint64_t count_for_bits = accu_bit_counts_for_scan_bases[shared_mem_index][column_index];
            uint32_t bit_count_offset = bit_counts_scan_bases[shared_mem_index][column_index];
            uint8_t bit_type = bit_types_for_scan_bases[shared_mem_index][column_index];
        
            uint64_t query_bit = query[0] & base; //((uint64_t)1 << column_index); //base;
        
            // the index for currently uncompressing binary code
            uint64_t current_binary_index = 0;
        
            while (count_for_bits <= max_length[0]) {
                count_for_bits += decode_VLQ_base64(vlq_bit_counts[column_index], &bit_count_offset); 
        
                if ((count_for_bits > max_length[0]) || (current_binary_index > batch_size))
                    break;
        
                while ((count_for_bits > (i_for_batch + current_binary_index)) && (current_binary_index < batch_size) && (i_for_batch + current_binary_index < max_length[0])) {
                    if (bit_type == 1) {
                        if (query_bit == 0)
                            local_distances[current_binary_index]++;
                    } else {
                        if (query_bit > 0)
                            local_distances[current_binary_index]++;
                    }
                    // move to next binary code
                    current_binary_index++;
                } 
                bit_type = bit_type ^ 0x01;
            } 
            base = base << 1;
        } 

        for (uint64_t distance_idx = 0; distance_idx < batch_size; distance_idx++) {
            distances[i_for_batch + distance_idx] = local_distances[distance_idx];
        }
    }
}
        """)
        
        self.vlq_compressed_hamming_dist = self.vlq_compressed_mod.get_function("vlq_compressed_hamming_dist")
 
        self.compressed_mod = SourceModule("""
#define BLOCK_SIZE 256

typedef unsigned int uint8_t;
typedef unsigned long int uint32_t;
typedef unsigned long long int uint64_t;
__global__ void compressed_hamming_dist(uint64_t* query, uint64_t** bit_counts, uint64_t* max_length, uint64_t* distances)
{
    const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    
    // how many binary codes we uncompress in a cuda thread
    int batch_size = 50;
    
    uint64_t local_distances[50];
    
    const uint64_t i_for_batch = i * batch_size;

    // pre-scan bit count matrix to skip redundant scaning for other threads in same block

    // 64 columns; for each column, record the base index as the beginning for threads in same block to scan
    __shared__ uint32_t bit_counts_scan_bases[BLOCK_SIZE / 64][64];
    __shared__ uint8_t bit_types_for_scan_bases[BLOCK_SIZE / 64][64];
    __shared__ uint64_t accu_bit_counts_for_scan_bases[BLOCK_SIZE / 64][64];


    // shared memory index: 64 threads share common block of shared memory
    uint8_t shared_mem_index = threadIdx.x / 64;

    // column id that this thread is responsible to find beginning base
    uint8_t thread_column_index = threadIdx.x % 64;

    if (thread_column_index < 64) {

        uint32_t smallest_binary_code_index = (gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + 64 * shared_mem_index) * batch_size;

        uint64_t count_for_bits = 0;
        uint64_t bit_count_index = 0;
        uint8_t bit_type = 0x00;
        while (count_for_bits <= max_length[0]) {
            count_for_bits += bit_counts[thread_column_index][bit_count_index];

            if (count_for_bits > smallest_binary_code_index) {
                bit_counts_scan_bases[shared_mem_index][thread_column_index] = bit_count_index;
                bit_types_for_scan_bases[shared_mem_index][thread_column_index] = bit_type;
                accu_bit_counts_for_scan_bases[shared_mem_index][thread_column_index] = count_for_bits - bit_counts[thread_column_index][bit_count_index];
                break;
            }

            bit_count_index++;
            bit_type = bit_type ^ 0x01;                
        }

    }

    // wait for other threads to find scan bases
    __syncthreads();

    
    if (i_for_batch < max_length[0]) {
    
        for (uint64_t distance_idx = 0; distance_idx < batch_size; distance_idx++) {
            local_distances[distance_idx] = 0;
        }
        
        uint64_t base = 1;
        
        for (uint64_t column_index = 0; column_index < 64; column_index++) {
        
            uint64_t count_for_bits = accu_bit_counts_for_scan_bases[shared_mem_index][column_index];
            uint64_t bit_count_index = bit_counts_scan_bases[shared_mem_index][column_index];
            uint8_t bit_type = bit_types_for_scan_bases[shared_mem_index][column_index];
        
            uint64_t query_bit = query[0] & base; //((uint64_t)1 << column_index); //base;
        
            // the index for currently uncompressing binary code
            uint64_t current_binary_index = 0;
        
            while (count_for_bits <= max_length[0]) {
                count_for_bits += bit_counts[column_index][bit_count_index++];
        
                if ((count_for_bits > max_length[0]) || (current_binary_index > batch_size))
                    break;
        
                while ((count_for_bits > (i_for_batch + current_binary_index)) && (current_binary_index < batch_size) && (i_for_batch + current_binary_index < max_length[0])) {
                    if (bit_type == 1) {
                        if (query_bit == 0)
                            local_distances[current_binary_index]++;
                    } else {
                        if (query_bit > 0)
                            local_distances[current_binary_index]++;
                    }
                    // move to next binary code
                    current_binary_index++;
                } 
                bit_type = bit_type ^ 0x01;
            } 
            base = base << 1;
        } 

        for (uint64_t distance_idx = 0; distance_idx < batch_size; distance_idx++) {
            distances[i_for_batch + distance_idx] = local_distances[distance_idx];
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

    def cuda_hamming_dist_in_compressed_domain(self, vec_a, compressed_columns, image_ids, vlq_mode):

        binary_code_length = len(image_ids)

        self.benchmark_begin('preparing')

        addresses = [] 
        gpu_alloc_objs = []
        for column in compressed_columns:
            #column_addr = drv.to_device(buffer(array.array('L', column), 0))
            column_addr = drv.to_device(column)
            gpu_alloc_objs.append(column_addr)
            addresses.append(int(column_addr))

        np_addresses = numpy.array(addresses).astype(numpy.uint64)

        # suppose we use 32 bit address space that 1 point costs 4 bytes
        # todo: do we have better way to figure the size of pointer in python?
        arrays_gpu = drv.mem_alloc(np_addresses.shape[0] * 8)

        drv.memcpy_htod(arrays_gpu, np_addresses)

        distances = numpy.zeros(binary_code_length).astype(numpy.uint64)

        #binary_code_length = 10000
 
        length = numpy.array([binary_code_length]).astype(numpy.uint64)
 
        print "total: " + str(binary_code_length) + " compressed binary codes." 

        self.benchmark_end('preparing')
        self.benchmark_begin('cudaing')

        if vlq_mode == 'n':
            print "non VLQ base64 cuda uncompression and hamming distance calculation"
            self.compressed_hamming_dist(
                drv.In(vec_a), arrays_gpu, drv.In(length), drv.Out(distances),
                block = self.block, grid = self.grid)
        else:
            print "VLQ base64 cuda uncompression and hamming distance calculation"            
            self.vlq_compressed_hamming_dist(
                drv.In(vec_a), arrays_gpu, drv.In(length), drv.Out(distances),
                block = self.block, grid = self.grid)

        self.benchmark_end('cudaing')

        print distances
        count = 0
        for dis in distances:
            print "count: " + str(count) + " " + str(dis) + " image: " + str(image_ids[count])
            count += 1
        print distances.shape

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
 
        self.hamming_dist(
                drv.In(vec_a), drv.InOut(dest), drv.In(length),
                block = self.block, grid = self.grid)
        
        print dest
        #for d in dest:
        #    print d
        print dest.shape
        
        return dest


