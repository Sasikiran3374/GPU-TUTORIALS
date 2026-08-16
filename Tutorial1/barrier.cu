#include <stdio.h>
#include <cuda_runtime.h>

__device__ int count = 0;

__global__ void kernel()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    printf("before barrier: thread %d\n", gid);

    // Sync threads within the block
    __syncthreads();

    // Each thread reaches the barrier
    atomicAdd(&count, 1);


    int totalThreads = gridDim.x * blockDim.x;

    // Wait until all threads reach the barrier
    while (atomicAdd(&count, 0) < totalThreads)
    {
        // Wait
    }

    // sync threads within the block again 
    // this is optional for this code beacause at while loop we are ensuring safely all threads completed execution or not
    __syncthreads();
    printf("after barrier: thread %d\n", gid);
}

int main()
{
    kernel<<<2, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}
