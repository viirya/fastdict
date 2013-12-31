
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <vector>


typedef unsigned int uint8_t;
typedef unsigned int uint16_t;
typedef unsigned long int uint32_t;
typedef unsigned long long int uint64_t;
 
__global__ void compressed_hamming_dist(uint64_t query, uint64_t** bit_counts, uint64_t max_length, uint64_t* distances)
{
  const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t xor_r;

  if (i < max_length) {

    uint64_t binary_code = 0x00;
    uint64_t mark_bit = 0x01;

    for (int column_index = 0; column_index < 64; column_index++) {

        uint32_t count_for_bits = 0;
        uint32_t bit_count_index = 0;
        uint8_t bit_type = 0x00;

        while (count_for_bits < max_length) {
            count_for_bits += bit_counts[column_index][bit_count_index++];
            if (count_for_bits > i) {
                if (bit_type == 1)
                    binary_code = binary_code | (mark_bit << column_index);
                break;
            }
            bit_type = bit_type ^ 0x01;
        }
    } 


    xor_r = query ^ binary_code;

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

    //distances[i] = binary_code;
    distances[i] = (xor_r * h01) >> 56;
    //distances[i] = i;
  }
}
 

uint64_t* gpu_hamming_dist_in_compressed_domain(uint64_t query, std::vector<std::vector<uint32_t> >& columns, uint64_t max_length)  {

    
    const int N_ARRAYS = 64;

    void* h_array = malloc(sizeof(uint64_t*) * N_ARRAYS);
    for(int i = 0; i < N_ARRAYS; i++) {
        cudaMalloc((void**)(((uint64_t*)h_array) + i * 8), columns[i].size() * sizeof(uint64_t));
    }
    void* d_array = NULL;
    cudaMalloc(&d_array, sizeof(void*) * N_ARRAYS);

    void* distances;
    cudaMalloc(&distances, sizeof(uint64_t) * max_length);

    cudaMemcpy(d_array, h_array, sizeof(void*) * N_ARRAYS, cudaMemcpyHostToDevice);

    dim3 BlockDim = dim3(50, 1, 1);
    dim3 GridDim  = dim3(50, 10, 1);

    compressed_hamming_dist<<<GridDim, BlockDim>>>(query, (uint64_t**)d_array, max_length, (uint64_t*)distances);
    cudaThreadSynchronize();

    void* h_distance_array = malloc(sizeof(uint64_t) * max_length);
    cudaMemcpy(h_distance_array, distances, max_length, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(distances);
 
    for(int i = 0; i < N_ARRAYS; i++) {
        cudaFree(((uint64_t*)h_array) + i * 8);
    }
 
    free(h_array);

    return (uint64_t*)h_distance_array;

}


