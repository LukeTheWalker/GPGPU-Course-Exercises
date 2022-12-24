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

__global__ void init (int *d_a, int N, int M){
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    if (r >= N || c >= M) return;
    d_a[r * M + c] = r - c;
}

void init_mat(int *d_a, int N, int M, int lws){
    int row_blks = (N + lws - 1) / lws;
    int col_blks = (M + lws - 1) / lws;

    dim3 blk_num;
    blk_num.x = col_blks;
    blk_num.y = row_blks;
    
    dim3 blk_size;
    blk_size.x = lws;
    blk_size.y = lws;

    printf("row_blks: %d, col_blks: %d\n", row_blks, col_blks);

    init<<<blk_num, blk_size>>>(d_a, N, M);
    cudaError_t err = cudaGetLastError();
    cuda_err_check(err, __FILE__, __LINE__);
}

void verify_prof(int *array, int nrows, int ncols)
{
	for (int r = 0; r < nrows; ++r) {
		for (int c = 0; c < ncols; ++c) {
			int a = array[r*ncols+c];
			int expected = r - c;
#if 0
			if (r < 8 && c < 8)
				printf("%d\t", a);
#endif
			if (a != expected)
				fprintf(stderr, "mismatch @ %d, %d: %d != %d\n",
					r, c, a, expected);
		}
		// if (r < 8) printf("\n");
	}
}

void print_mat(int *array, int nrows, int ncols)
{
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            int a = array[r*ncols+c];
            printf("%d\t", a);
        }
        printf("\n");
    }
}

void verify (int *h_a, int N, int M){
    for (int r = 0; r < N; ++r) {
		for (int c = 0; c < M; ++c) {
			int a = h_a[r*M+c];
			int expected = r - c;
            if (a != expected)
				fprintf(stderr, "mismatch @ %d, %d: %d != %d\n", r, c, a, expected);
		}
    }
}

int main (int argc, char * argv[]){
    if (argc != 4){
        printf("Usage: %s <N> <M> <lws>", argv[0]);
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int lws = atoi(argv[3]);

    int *h_a, *d_a;
    float t1;
    cudaError_t err;

    size_t memsize = N*M*sizeof(int);

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

    init_mat(d_a, N, M, lws);

    err = cudaEventRecord(post_init);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(h_a, d_a, memsize, cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventRecord(post_memcpy);
    cuda_err_check(err, __FILE__, __LINE__);

    // verify(h_a, N, M);
    printf("------------------------\n");
    verify_prof(h_a, N, M);
    // print_mat(h_a, N, M);

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