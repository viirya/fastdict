import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
 

def cuda_hamming_dist(vec_a, vec_b):

 
    mod = SourceModule("""
typedef unsigned long long int uint64_t;
__global__ void hamming_dist(uint64_t *dest, uint64_t *a, uint64_t *b, uint64_t *length)
{
  const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t xor_r;

  if (i < %(length)s) {
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

    dest[i] = (xor_r * h01) >> 56;
  }

}
""" % {"length": vec_b.shape[0]})

    hamming_dist = mod.get_function("hamming_dist")

    dest = numpy.zeros_like(vec_b)
    length = numpy.array([vec_b.shape[0]]).astype(numpy.uint64)
    hamming_dist(
            drv.Out(dest), drv.In(vec_a), drv.In(vec_b), drv.In(length),
            block=(500, 1, 1), grid=(500, 200))

    print dest

    return dest


