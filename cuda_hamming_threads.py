
import pycuda
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import threading
import numpy

class GPUThread(threading.Thread):
    def __init__(self, number, vec_a, vec_b, block, grid):
        threading.Thread.__init__(self)

        self.number = number
        self.vec_a = vec_a
        self.vec_b = vec_b

        self.block = block
        self.grid = grid

    def run(self):
        self.dev = drv.Device(self.number)
        self.ctx = self.dev.make_context()
        self.ctx.push()

        self.vec_b = self.multi_iteration(self.vec_a, self.vec_b)
        #self.vec_b = hamming_kernel(self.vec_a, self.vec_b, self.block, self.grid)

        print "successful exit from thread %d" % self.number

        self.ctx.pop()
        self.ctx.detach()
 
    def multi_iteration(self, vec_a, vec_b):

        vector_len = vec_b.shape[0]
 
        sections = range(0, vector_len, 10000000)
        sections = sections[1:]

        sub_vec_bs = numpy.split(vec_b, sections)

        dest = numpy.array([])
        for sub_vec in sub_vec_bs:
            #sub_dest = self.cuda_hamming_dist(vec_a, sub_vec)
            sub_dest = hamming_kernel(self.vec_a, sub_vec, self.block, self.grid)
            dest = numpy.concatenate((dest, sub_dest))

        return dest
 
def hamming_kernel(vec_a, vec_b, block, grid):
 
    vector_len = 100000
    
    mod = SourceModule("""
typedef unsigned long long int uint64_t;
__global__ void hamming_dist(uint64_t *a, uint64_t *b, uint64_t *length)
{
  const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t xor_r;

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
""")

    hamming_dist = mod.get_function("hamming_dist")

    dest = numpy.array(vec_b)    
    length = numpy.array([vec_b.shape[0]]).astype(numpy.uint64)
    
    hamming_dist(
            drv.In(vec_a), drv.InOut(dest), drv.In(length),
            block = block, grid = grid)
    
    print dest
    
    return dest
 
class CudaHamming(object):

    def __init__(self, block = (500, 1, 1), grid = (500, 200)):

        self.block = block
        self.grid = grid             

    def run_kernel_on_gpus(self, vec_a, vec_b):

        drv.init()
        num = drv.Device.count()
        num = 1

        vector_len = vec_b.shape[0]

        sections = range(0, vector_len, vector_len / num)
        sections = sections[1:]
        print "section on gpus:"
        print sections

        sub_vec_bs = numpy.split(vec_b, sections)

        gpu_thread_list = []
        for i in range(num):
            gpu_thread = GPUThread(i, vec_a, sub_vec_bs[i], self.block, self.grid)
            gpu_thread.start()
            gpu_thread_list.append(gpu_thread)

        dest = numpy.array([])
        for gpu in gpu_thread_list:
            gpu.join()
            dest = numpy.concatenate((dest, gpu.vec_b))

        print dest

        return dest

