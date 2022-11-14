#include <stdio.h>
#include <stdlib.h>

template <int BLOCK_SIZE>
__global__ void matrix_mul(float *dest, float *srcL, float *srcR, int n, int m, int k)
{
    // srcL(n * m), srcR(m * k), dest(n * k)

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int srcL_begin = BLOCK_SIZE * bx * m;
    int srcL_end = srcL_begin + m;
    int srcL_step = BLOCK_SIZE;

    int srcR_begin = BLOCK_SIZE * by;
    int srcR_step = BLOCK_SIZE * k;

    float dest_sub = 0;

    __shared__ float srcL_s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float srcR_s[BLOCK_SIZE][BLOCK_SIZE];
    for (int idxL = srcL_begin, idxR = srcR_begin; idxL < srcL_end; idxR += srcR_step, idxL += srcL_step)
    {
        srcL_s[tx][ty] = srcL[idxL + tx * m + ty];
        srcR_s[tx][ty] = srcR[idxR + tx * k + ty];

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            dest_sub += srcL_s[tx][i] * srcR_s[i][ty];
        }

        __syncthreads();
    }

    int block_dest_idx = BLOCK_SIZE * bx * m + BLOCK_SIZE * by;
    dest[block_dest_idx + m * tx + ty] = dest_sub;
}

#define N 768
#define M 768
#define K 768
#define BLOCK_SIZE 32
#define RNR 4

int main()
{
    float h_M1[N * M], h_M2[M * K], h_R[N * K];
    
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            h_M1[i * M + j] = (float)rand() / (float)(RAND_MAX / RNR);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            h_M2[i * K + j] = (float)rand() / (float)(RAND_MAX / RNR);

    float *d_M1, *d_M2, *d_R;
    cudaMalloc((void **)&d_M1, N * M * sizeof(float));
    cudaMalloc((void **)&d_M2, M * K * sizeof(float));
    cudaMalloc((void **)&d_R, N * K * sizeof(float));

    cudaMemcpy(d_M1, h_M1, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, h_M2, M * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N / threads.x, M / threads.y);
    matrix_mul <BLOCK_SIZE> <<<grid, threads>>>(d_R, d_M1, d_M2, N, M, K);

    cudaMemcpy(h_R, d_R, N * K * sizeof(float), cudaMemcpyDeviceToHost);

    float t;

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            t = 0;
            for (int k = 0; k < K; ++k)
            {
                t += h_M1[i * M + k] * h_M2[k * K + j];
            }

            if (t - h_R[i * K + j] > 0.001 || t - h_R[i * K + j] < -0.001)
            {
                printf("Failed! --- %d, %d --- %f ___ %f\n", i, j, t, h_R[i * K + j]);
                return 0;
            }
        }
    }

    printf("Success!\n");

    return 0;
}