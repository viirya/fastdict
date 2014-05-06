import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import sys

mod = SourceModule("""
typedef unsigned long long int uint64_t;
typedef unsigned int uint8_t;
__global__ void diag_kernel(char** dest) //uint64_t* a, uint64_t* b, uint64_t* c)
{

/*
    a[0] = 1;
    b[0] = 2;
    c[0] = 3;

    b[1] = 4;
    c[1] = 4;
*/
    dest[0][0] = 1;
    dest[1][0] = 1;
    dest[1][1] = 2; 
    dest[1][2] = 3;
    dest[2][0] = 1;
    //uint32_t* arrays[3] = {a, b, c};
    //arrays[1][0] = 3;
}
""")

diag_kernel = mod.get_function("diag_kernel")

a1 = np.zeros(1, dtype=np.uint8)
b1 = np.zeros(10, dtype=np.uint8)
c1 = np.zeros(100, dtype=np.uint8)

print a1
print b1
print c1

a1_addr = drv.to_device(a1)
b1_addr = drv.to_device(b1)
c1_addr = drv.to_device(c1)

#print int(a1_addr)
#print sys.getsizeof(int(b1_addr)) 

twod_gpu = drv.mem_alloc(3 * 8)
address = np.array([int(a1_addr), int(b1_addr), int(c1_addr)]).astype(np.uint64)
#print address

drv.memcpy_htod(twod_gpu, address)

#diag_kernel(drv.InOut(a1), drv.InOut(b1), drv.InOut(c1), block=(32,1,1))
diag_kernel(twod_gpu, block=(32,1,1))

drv.memcpy_dtoh(a1, a1_addr)
print a1
drv.memcpy_dtoh(b1, b1_addr)
print b1
drv.memcpy_dtoh(c1, c1_addr)
print c1
 
