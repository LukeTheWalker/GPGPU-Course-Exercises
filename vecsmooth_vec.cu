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

__global__ void smooth_v4 (int4 * d_in, int4 * d_out, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int4 sum = d_in[idx];
    int4 prev = {0, sum.x, sum.y, sum.z};
    int4 next = {sum.y, sum.z, sum.w, 0};
    int4 div = make_int4(2, 3, 3, 2);
    if (idx > 0  ) {prev.x = d_in[idx-1].w; div.x++;}
    if (idx < n-1) {next.w = d_in[idx+1].x; div.w++;}
    d_out[idx] = {(sum.x + prev.x + next.x) / div.x,
                  (sum.y + prev.y + next.y) / div.y,
                  (sum.z + prev.z + next.z) / div.z,
                  (sum.w + prev.w + next.w) / div.w};
}

// DOMANDA: cuda non supporta int8 o int16?
// __global__ void smooth_v8(int8 *d_in, int8 *d_out, int n){
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int8 sum = d_in[idx];
//     int8 prev = {0, sum.x, sum.y, sum.z, sum.w, 0, 0, 0};
//     int8 next = {sum.y, sum.z, sum.w, sum.x, sum.y, sum.z, sum.w, 0};
//     int8 div = make_int8(2, 3, 3, 3, 3, 2, 2, 2);
//     if (idx > 0  ) {prev.x = d_in[idx-1].w; div.x++;}
//     if (idx < n-1) {next.w = d_in[idx+1].x; div.w++;}
//     d_out[idx] = {(sum.x + prev.x + next.x) / div.x,
//                   (sum.y + prev.y + next.y) / div.y,
//                   (sum.z + prev.z + next.z) / div.z,
//                   (sum.w + prev.w + next.w) / div.w,
//                   (sum.x + prev.x + next.x) / div.x,
//                   (sum.y + prev.y + next.y) / div.y,
//                   (sum.z + prev.z + next.z) / div.z,
//                   (sum.w + prev.w + next.w) / div.w};
// }

void smooth_array_v4(int * d_in, int * d_out, int n){
    n /= 4;
    int lws = 256;
    int nblks = (n + lws - 1) / lws;
    smooth_v4<<<nblks, lws>>>((int4*)d_in, (int4*)d_out, n);
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

    smooth_array_v4(d_in, d_out, nels);

    err = cudaEventRecord(post_smooth);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(h_a, d_out, memsize, cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventRecord(post_memcpy);
    cuda_err_check(err, __FILE__, __LINE__);

    verify(h_a, nels);

    err = cudaEventRecord(post_verify);
    cuda_err_check(err, __FILE__, __LINE__);

    // DOMANDA: perché nella veraione OpenCL moltiplichiamo memsize per 2?
    cudaEventElapsedTime(&t1, pre_init, post_init);
    printf("init : %f ms (%f GB/s)\n", t1, memsize/t1/1e6);

    // DOMANDA: perché moltiplichiamo per 1.5 e abbiamo un uso della bandwidth terribile?
    cudaEventElapsedTime(&t1, post_init, post_smooth);
    printf("smooth : %f ms (%f GB/s)\n", t1, 1.5*memsize/t1/1e6);

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