
#include "common.h"
#include "timer.h"

__global__ void mm_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if(col < N &&row < M){

    for(unsigned int i = 0; i < K; i++){
        //CHECK IF i IS OUT OF BOUND
        sum += A[row*K + i]*B[i*N + col]
    }
    c[row*N + col] = sum;
    }


}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, M*K*sizeof(float));
    cudaMalloc((void**) &B_d, K*N*sizeof(float));
    cudaMalloc((void**) &C_d, M*N*sizeof(float));




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(A_d, A, M*K*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(B_d, B, K*N*sizeof(float), cudaMemcpyHostToDevice );





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks((N + numThreadsPerBlock.x -1 )/numThreadsPerBlock.x, (M + numThreadsPerBlock.x -1 )/numThreadsPerBlock.x);
    mm_kernel <<< numBlocks, numThreadsPerBlock>>> (A_d, B_d, C_d,M,N,K);




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(C, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

