#include <stdio.h>
#include <map>
#include <vector>
#include <string>

__host__ __device__ int round_div_up (int a, int b){
    return (a + b - 1)/b;
}

__host__ __device__ int round_mul_up (int a, int b){
    return round_div_up(a, b)*b;
}

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

std::map<std::string, cudaEvent_t> create_events (std::vector<std::string> events){
    std::map<std::string, cudaEvent_t> event_list;
    cudaError_t err;
    for (auto event : events){
        cudaEvent_t e;
        err = cudaEventCreate(&e);
        cuda_err_check(err, __FILE__, __LINE__);
        event_list.insert(std::make_pair(event, e));
    }
    return event_list;
}


template<typename T>
void print_array (T * d_a, int n) {
    for (int i = 0; i < n; i++){
        T tail;
        cudaError_t err = cudaMemcpy(&tail, d_a+i, sizeof(T), cudaMemcpyDeviceToHost);
        cuda_err_check(err, __FILE__, __LINE__);
        printf("%u ", tail);
    }
}