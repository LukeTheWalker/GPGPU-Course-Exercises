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

__global__ void init (int *d_a, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    d_a[idx] = idx;
}

void init_array(int *d_a, int n){
    int lws = 256;
    int nblks = (n + lws - 1) / lws;
    init<<<nblks, lws>>>(d_a, n);
}


__global__ void smooth_lmem (int * d_in, int * d_out, int n){
    extern __shared__ int lmem[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int lidx = threadIdx.x;
    if (idx >= n) return;

    int div = 1;

    lmem[lidx] = d_in[idx];

    __syncthreads();

    int sum = lmem[lidx];
    if (idx > 0) {
        if (lidx != 0) 
            sum += lmem[lidx - 1];
        else 
            sum += d_in[idx - 1]; 
        div++;
    }
    if (idx < n-1) {
        if (lidx != blockDim.x - 1) 
            sum += lmem[lidx + 1];
        else 
            sum += d_in[idx + 1]; 
        div++;
    }
    d_out[idx] = sum / div;
}

__global__ void smooth_lmem_oversize (int * d_in, int * d_out, int n){
    extern __shared__ int lmem[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int lidx = threadIdx.x;
    int * lmem_ptr = ((int*)(&lmem))+1;
    if (idx >= n) return;

    int div = 1;

    lmem_ptr[lidx] = d_in[idx];
    if (lidx == 0 && idx != 0) 
        lmem_ptr[lidx - 1] = d_in[idx - 1];
    if (lidx == blockDim.x - 1 && idx != n - 1)
        lmem_ptr[lidx + 1] = d_in[idx + 1];

    __syncthreads();

    int sum = lmem_ptr[lidx];

    if (idx > 0) {
        sum += lmem_ptr[lidx - 1];
        div++;
    }
    if (idx < n-1) {
        sum += lmem_ptr[lidx + 1];
        div++;
    }

    d_out[idx] = sum / div;
}

void smooth_array(int * d_in, int * d_out, int n){
    int lws = 256;
    int nblks = (n + lws - 1) / lws;
#if 1
    // DOMANDA: prima mi sono dimenticato di aggiungere 2 alla lws, non è crashato, perché?
    smooth_lmem_oversize<<<nblks, lws, (lws+2) * sizeof(int)>>>(d_in, d_out, n);
#else
    smooth_lmem<<<nblks, lws, lws * sizeof(int)>>>(d_in, d_out, n);
#endif
}

void verify (int *h_a, int nels){
    for (int i = 0; i < nels; ++i) {
        int expected = i - !!(i == nels - 1);

        if (h_a[i] != expected) {
            fprintf(stderr, "mismatch @ %d: %d != %d\n",
            i, h_a[i], expected);
        }
    }
}

int main (int argc, char * argv[]){
    if (argc != 2){
        printf("Usage: %s <nels>", argv[0]);
    }
    int nels = atoi(argv[1]);
    int *h_a, *d_in, *d_out;
    float t1;
    cudaError_t err;

    size_t memsize = nels*sizeof(int);

    cudaEvent_t pre_init, post_init, post_smooth, post_memcpy, post_verify;
    err = cudaEventCreate(&pre_init);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventCreate(&post_init);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventCreate(&post_smooth);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventCreate(&post_memcpy);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventCreate(&post_verify);
    cuda_err_check(err, __FILE__, __LINE__);

    //  finished with event creation
    
    err = cudaMalloc    (&d_in, memsize);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMalloc    (&d_out, memsize);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMallocHost(&h_a, memsize);
    cuda_err_check(err, __FILE__, __LINE__);

    // finished with memory allocation

    err = cudaEventRecord(pre_init);
    cuda_err_check(err, __FILE__, __LINE__);

    init_array(d_in, nels);

    err = cudaEventRecord(post_init);
    cuda_err_check(err, __FILE__, __LINE__);

    smooth_array(d_in, d_out, nels);

    err = cudaEventRecord(post_smooth);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(h_a, d_out, memsize, cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventRecord(post_memcpy);
    cuda_err_check(err, __FILE__, __LINE__);

    verify(h_a, nels);

    err = cudaEventRecord(post_verify);
    cuda_err_check(err, __FILE__, __LINE__);

    cudaEventElapsedTime(&t1, pre_init, post_init);
    printf("init : %f ms (%f GB/s)\n", t1, memsize/t1/1e6);

    cudaEventElapsedTime(&t1, post_init, post_smooth);
    printf("smooth : %f ms (%f GB/s)\n", t1, 4*memsize/t1/1e6);

    cudaEventElapsedTime(&t1, post_smooth, post_memcpy);
    printf("memcp: %f ms (%f GB/s)\n", t1, memsize/t1/1e6);

    cudaEventElapsedTime(&t1, post_memcpy, post_verify);
    printf("check: %f ms\n", t1);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_a);
    cudaEventDestroy(pre_init);
    cudaEventDestroy(post_init);
    cudaEventDestroy(post_smooth);
    cudaEventDestroy(post_memcpy);
    cudaEventDestroy(post_verify);
}