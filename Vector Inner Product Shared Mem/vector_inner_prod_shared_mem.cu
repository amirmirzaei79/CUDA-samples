#include <stdlib.h>

#define ARRAY_LEN 1024
#define RAND_LIM 2000

__global__ void array_internal_mul(int *src1, int *src2, int *dest, int length)
{
    int block_start = length * blockIdx.x / gridDim.x;
    int block_end = length * (blockIdx.x + 1) / gridDim.x;

    int i;
    int r;
    
    __shared__ int shared_result;
    if (threadIdx.x == 0)
    	shared_result = 0;

   for (i = block_start; i < block_end; i += blockDim.x)
    {
        if (threadIdx.x + i < block_end)
        {
            r = src1[threadIdx.x + i] * src2[threadIdx.x + i];
            atomicAdd(&shared_result, r);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(dest, shared_result);
}

int main()
{
    srand(time(0));

    int source1[ARRAY_LEN], source2[ARRAY_LEN], destination = 0;
    int i;
    const int array_size = ARRAY_LEN * sizeof(int);

    for (i = 0; i < ARRAY_LEN; ++i)
        source1[i] = (rand() % RAND_LIM) - (RAND_LIM / 2);
    for (i = 0; i < ARRAY_LEN; ++i)
        source2[i] = (rand() % RAND_LIM) - (RAND_LIM / 2);

    int *d_src1, *d_src2, *d_dest;
    cudaMalloc((void **)&d_src1, array_size);
    cudaMalloc((void **)&d_src2, array_size);
    cudaMalloc((void **)&d_dest, sizeof(int));

    cudaMemcpy(d_src1, source1, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src2, source2, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, &destination, sizeof(int), cudaMemcpyHostToDevice);

    const int block_size = 256;
    // const int grid_size = ARRAY_LEN / block_size;
    const int grid_size = 2;

    array_internal_mul <<<grid_size, block_size>>> (d_src1, d_src2, d_dest, ARRAY_LEN);

    cudaMemcpy(&destination, d_dest, sizeof(int), cudaMemcpyDeviceToHost);

    return 0;
}
