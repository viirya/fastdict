import pycuda.autoinit
import pycuda.driver as drv
import numpy
import array
import time

from pycuda.compiler import SourceModule

class CudaHamming(object):

    def __init__(self,  block = (256, 1, 1), grid = (10, 1)):

        vector_len = 100000

        self.compressed_mod = SourceModule("""
typedef unsigned int uint8_t;
typedef unsigned long int uint32_t;
typedef unsigned long long int uint64_t;
__global__ void compressed_hamming_dist(uint64_t* query, uint64_t** bit_counts, uint64_t* max_length, uint64_t* distances)
{
  const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;

  // how many binary codes we uncompress in a cuda thread
  int batch_size = 1;

  const uint64_t i_for_batch = i * batch_size;

  if (i_for_batch < max_length[0]) {

    for (uint64_t distance_idx = 0; distance_idx < batch_size; distance_idx++) {
        distances[i_for_batch + distance_idx] = 0;
    }

    uint64_t base = 1;
    
    for (uint64_t column_index = 0; column_index < 64; column_index++) {

        uint64_t count_for_bits = 0;
        uint64_t bit_count_index = 0;
        uint8_t bit_type = 0x00;

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
                        distances[i_for_batch + current_binary_index]++;
                } else {
                    if (query_bit > 0)
                        distances[i_for_batch + current_binary_index]++;
                }
                // move to next binary code
                current_binary_index++;
            } 
            bit_type = bit_type ^ 0x01;
        } 
        base = base << 1;
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

  //if (i < %(length)s) {
  if (i < length[0]) {
    xor_r = a[0] ^ b[i];

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

    b[i] = (xor_r * h01) >> 56;
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

    def cuda_hamming_dist_in_compressed_domain(self, vec_a, compressed_columns, image_ids):

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
        self.compressed_hamming_dist(
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


