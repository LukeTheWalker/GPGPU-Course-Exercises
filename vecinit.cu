#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
        exit (EXIT_FAILURE);
    }
}

__global__ void init (int *d_a){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_a[idx] = idx;
}

void init_array(int *d_a, int n){
    int lws = 256;
    int nblks = (n + lws - 1) / lws;
    init<<<nblks, lws>>>(d_a);
}

void verify (int *h_a, int nels){
    for (int i = 0; i < nels; i++){
        if (h_a[i] != i)
            printf("Error at %d: %d\n", i, h_a[i]);
    }
}

int main (int argc, char * argv[]){
    if (argc != 2){
        printf("Usage: %s <nels>", argv[0]);
    }
    int nels = atoi(argv[1]);
    int *h_a, *d_a;
    float t1;
    cudaError_t err;

    size_t memsize = nels*sizeof(int);

    cudaEvent_t pre_init, post_init, post_memcpy, post_verify;
    err = cudaEventCreate(&pre_init);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventCreate(&post_init);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventCreate(&post_memcpy);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventCreate(&post_verify);
    cuda_err_check(err, __FILE__, __LINE__);

    //  finished with event creation
    
    err = cudaMalloc    (&d_a, memsize);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMallocHost(&h_a, memsize);
    cuda_err_check(err, __FILE__, __LINE__);

    // finished with memory allocation

    err = cudaEventRecord(pre_init);
    cuda_err_check(err, __FILE__, __LINE__);

    init_array(d_a, nels);

    err = cudaEventRecord(post_init);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(h_a, d_a, memsize, cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventRecord(post_memcpy);
    cuda_err_check(err, __FILE__, __LINE__);

    verify(h_a, nels);

    err = cudaEventRecord(post_verify);
    cuda_err_check(err, __FILE__, __LINE__);

    cudaEventElapsedTime(&t1, pre_init, post_init);
    printf("init : %f ms (%f GB/s)\n", t1, memsize/t1/1e6);

    cudaEventElapsedTime(&t1, post_init, post_memcpy);
    printf("memcp: %f ms (%f GB/s)\n", t1, memsize/t1/1e6);

    cudaEventElapsedTime(&t1, post_memcpy, post_verify);
    printf("check: %f ms\n", t1);

    cudaFree(d_a);
    cudaFreeHost(h_a);
    cudaEventDestroy(pre_init);
    cudaEventDestroy(post_init);
    cudaEventDestroy(post_memcpy);
    cudaEventDestroy(post_verify);
}