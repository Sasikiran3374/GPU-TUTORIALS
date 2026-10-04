
#include <iostream>
#include <cuda.h>
#include <mma.h>

using namespace std;
using namespace nvcuda;

#define N 64
#define TILE 16

__global__ void tensorCoreMatMul(const float *A,const float *B,float *C)
{
    // One warp computes one 16x16 output tile
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    // Tensor Core fragments
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE,half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE,float> c_frag;

    // Start accumulator with zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Temporary 16x16 tiles
    __shared__ half A_tile[TILE][TILE];
    __shared__ half B_tile[TILE][TILE];

    int lane = threadIdx.x;

    // There are 4 tiles along K dimension
    for (int k = 0; k < N; k += TILE)
    {
        // Load A and B tiles into shared memory
        for (int i = lane; i < TILE * TILE; i += 32)
        {
            int row = i / TILE;
            int col = i % TILE;
            A_tile[row][col] = __float2half(A[(tileRow * TILE + row) * N + (k + col)]);
            B_tile[row][col] = __float2half(B[(k + row) * N + (tileCol * TILE + col)]);
        }
        __syncthreads();

        // Load tiles into Tensor Core fragments
        wmma::load_matrix_sync(a_frag, &A_tile[0][0],TILE);
        wmma::load_matrix_sync(b_frag, &B_tile[0][0],TILE);

        // Tensor Core matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Store the final 16x16 result
    wmma::store_matrix_sync(C + tileRow * TILE * N + tileCol * TILE,c_frag,N,wmma::mem_row_major);
}


// CPU matrix multiplication for verification
void cpuMatMul(const float *A,const float *B,float *C)
{
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float sum = 0.0f;
            for (int k = 0; k < N; k++){
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


int main()
{
    size_t size = N * N * sizeof(float);

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    float *h_C_cpu = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // One warp = one 16x16 output tile
    dim3 block(32);
    dim3 grid(N / TILE, N / TILE);

    // Run Tensor Core multiplication
    tensorCoreMatMul<<<grid, block>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cpuMatMul(h_A, h_B, h_C_cpu);

    // Verify result
    bool correct = true;

    for (int i = 0; i < N * N; i++){
        if (fabs(h_C[i] - h_C_cpu[i]) > 1e-2){
            correct = false;
            break;
        }
    }

    cout << "Matrix size : 64 x 64" << endl;
    cout << "Tile size   : 16 x 16" << endl;
    cout << "Output tiles: 4 x 4 = 16" << endl;
    cout << endl;

    if (correct)
        cout << "Tensor Core Matrix Multiplication: SUCCESS" << endl;
    else
        cout << "Result verification: FAILED" << endl;

    cout << endl;
    cout << "C[0][0] = " << h_C[0] << endl;
    cout << "Expected = " << h_C_cpu[0] << endl;

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;

    return 0;
}