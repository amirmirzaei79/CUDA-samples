#include <stdio.h>

#define N 1024 * 512

__global__ void vecAdd(int *dest, int *src1, int *src2, int length)
{
    int th_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (th_idx < length)
        dest[th_idx] = src1[th_idx] + src2[th_idx];
}

int main()
{
    int hA[N], hB[N], hC[N];
    int i;
    for (i = 0; i < N; ++i)
    {
        hA[i] = i;
        hB[i] = 2 * i;
        hC[i] = 0;
    }

    int *dA, *dB, *dC;
    const int size = sizeof(int) * N;
    cudaMalloc((void**)&dA, size);
    cudaMalloc((void**)&dB, size);
    cudaMalloc((void**)&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = N/block_size;

    vecAdd<<<grid_size, block_size>>> (dC, dA, dB, size);

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    int fail = 0;
    for (i = 0; i < N && !fail; ++i)
    {
        if (hA[i] + hB[i] != hC[i])
        {
            printf("Fail at index: %d\n", i);
            fail = 1;
            printf("Correct Value: %d\n", hA[i] + hB[i]);
        }
    }

    if (!fail)
    {
        printf("Success\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
