import pycuda.autoinit
import pycuda.driver as drv
import numpy
import array
import time

from pycuda.compiler import SourceModule

class CudaIndexing(object):

    def __init__(self,  block = (256, 1, 1), grid = (15, 1)):

        self.mod = SourceModule("""
typedef unsigned int uint8_t;
typedef unsigned long int uint32_t;
typedef unsigned long long int uint64_t;

__device__ uint8_t dot_product(double* source, unsigned char* target, int len) {

    double project = 0.0;
    for (int i = 0; i < len; i++) {
        project += (source[i] * (double)target[i]);
    }
    return project;
}

__global__ void indexing(double **planes, unsigned char *data_points, uint64_t *projects, uint64_t *length)
{
    const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;

    // how many data points we index in a cuda thread
    int batch_size = 50;
    
    const uint64_t i_for_batch = i * batch_size;

    if (i_for_batch < length[0]) {
        // for each data point
        for (uint64_t data_index = i_for_batch; data_index < i_for_batch + batch_size && data_index < length[0]; data_index++) {
            uint64_t binary_code = 0;
            uint64_t base = 0x8000000000000000;

            // project in 64 planes
            
            for (int plane_index = 0; plane_index < 64; plane_index++) {
                if (dot_product(planes[plane_index], (data_points + 128 * data_index), 128) > 0)
                    binary_code = binary_code | base;
                base = base >> (uint64_t)1;
            }
            
            projects[data_index] = binary_code;
        }
    }

}
""")

        self.indexing_kernel = self.mod.get_function("indexing")

        self.block = block
        self.grid = grid             

    def benchmark_begin(self, title):
        print "start to " + title
        self.start = time.clock()

    def benchmark_end(self, title):
        print "end of " + title
        elapsed = (time.clock() - self.start)
        print "time: " + str(elapsed)

    def batch_indexing(self, planes, data_points):

        data_size = data_points.shape[0] / 128

        self.benchmark_begin('preparing')

        gpu_alloc_objs = []

        # for data points

        #addresses = [] 
        #for point in data_points:
        #    point_addr = drv.to_device(point)
        #    gpu_alloc_objs.append(point_addr)
        #    addresses.append(int(point_addr))

        #np_addresses = numpy.array(addresses).astype(numpy.uint64)

        # 64 bit addressing space. each point costs 8 bytes
        #arrays_gpu = drv.mem_alloc(np_addresses.shape[0] * 8)
        #drv.memcpy_htod(arrays_gpu, np_addresses)

        # for planes

        planes_addresses = [] 
        for plane in planes:
            plane_addr = drv.to_device(plane)
            gpu_alloc_objs.append(plane_addr)
            planes_addresses.append(int(plane_addr))

        planes_np_addresses = numpy.array(planes_addresses).astype(numpy.uint64)

        # 64 bit addressing space. each point costs 8 bytes
        planes_arrays_gpu = drv.mem_alloc(planes_np_addresses.shape[0] * 8)
        drv.memcpy_htod(planes_arrays_gpu, planes_np_addresses)

        # projections
 
        projections = numpy.zeros(data_size).astype(numpy.uint64)

        length = numpy.array([data_size]).astype(numpy.uint64)
 
        print "total: " + str(data_size) + " data points to indexing." 

        self.benchmark_end('preparing')
        self.benchmark_begin('cudaing')

        self.indexing_kernel(
            planes_arrays_gpu, drv.In(data_points), drv.Out(projections), drv.In(length),
            block = self.block, grid = self.grid)
        
        self.benchmark_end('cudaing')

        #count = 0
        #for pro in projections:
        #    print "count: " + str(count) + " " + str(pro)
        #    count += 1
        #print projections.shape

        return projections



