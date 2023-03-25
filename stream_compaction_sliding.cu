#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <vector>
#include <string>
#include <map>

#include "utils.cu"

__global__ void compute_predicate(int4 *d_in, int4 *d_flags, int nquarts){
    int gi = threadIdx.x + blockIdx.x * blockDim.x;
    if (gi >= nquarts) return;
    int4 val = d_in[gi];
    d_flags[gi] = {val.x % 2 == 1, val.y % 2 == 1, val.z % 2 == 1, val.w % 2 == 1};
}

// scan of all elements in the workgroup window
__device__ int scan_single_element(
    int first_wg_el, 
    int end_wg_el, 
    int tail, 
    int4 *out, 
    int4 *in, 
    int *lmem
){
    int4 val = {0, 0, 0, 0};
    int li = threadIdx.x;
    int gi = first_wg_el + li;
    int lws = blockDim.x;
    // scan of single work-item (quart of int)
    if (gi < end_wg_el){
        val = in[gi];
        // val.s1 += val.s0 ; val.s3 += val.s2;
        val.y += val.x; val.w += val.z;
        // val.s2 += val.s1 ; val.s3 += val.s1;
        val.z += val.y; val.w += val.y;
    }

    // write work-item tail to local memory to sync with rest of workgroup
    lmem[li] = val.w;

    // scan of local memory
    for (int active_mask = 1; active_mask < lws; active_mask *= 2) {
		int pull_mask = active_mask - 1;
		pull_mask = ~pull_mask;
		pull_mask = li & pull_mask;
		pull_mask = pull_mask - 1;
		__syncthreads();
		if (li & active_mask) lmem[li] += lmem[pull_mask];
	}
    __syncthreads();

    // each work-item adds the previous work-item tail
    if (li > 0){
        val.x += lmem[li - 1]; 
        val.y += lmem[li - 1];
        val.z += lmem[li - 1]; 
        val.w += lmem[li - 1];
    }

    // each work-item adds the previous windows's tail
	val.x += tail; 
    val.y += tail;
    val.z += tail; 
    val.w += tail;

    // only write to global memory if the work-item is in the window
	if (gi < end_wg_el)
		out[gi] = val;

    // return the last element of the local memory
	return lmem[lws - 1];
}

__global__ void scan_sliding_window(
    int4 *d_in, 
    int4 *d_out,
    int *d_tails,
    int nquarts,
	int preferred_wg_multiple
){  
    extern __shared__ int lmem[];

    int nwg = gridDim.x;
    int wg_id = blockIdx.x;
    int lws = blockDim.x;

    // elemensts per workgroup
    int els_per_wg = round_div_up(nquarts, nwg);

    // optimize wavefront size
    els_per_wg = round_mul_up(els_per_wg, preferred_wg_multiple);

    // first and last element of the workgroup
    int first_el = wg_id*els_per_wg;
    int last_el = min(first_el + els_per_wg, nquarts);

    // get local id 
    int lid = threadIdx.x;

    // tail initialization
    int tail = 0;

    while (first_el < last_el) {
		tail += scan_single_element(first_el, last_el, tail, d_out, d_in, lmem);
		first_el += lws;
		__syncthreads();
	}
    
    if (nwg > 1 && lid == 0) {
		d_tails[wg_id] = tail;
	}
}

__global__ void scan_fixup (int4 *d_out, int *d_tails, int nquarts, int preferred_wg_multiple){
    const int nwg = gridDim.x;
	const int group_id = blockIdx.x;
    const int lws = blockDim.x;

	if (group_id == 0) return;

	// elements per work-group
	int els_per_wg = round_div_up(nquarts, nwg);
	els_per_wg = round_mul_up(els_per_wg, preferred_wg_multiple);

	// index of first element assigned to this work-group
	int first_el = els_per_wg*group_id;
	// index of first element NOT assigned to us
	const int end_el = min(nquarts, els_per_wg*(group_id+1));

	int fixup = d_tails[group_id-1];
	int gi = first_el + threadIdx.x;
	while (gi < end_el) {
        // printf("Before fixup: %d, %d, %d, %d, gi: %d\n", d_out[gi].x, d_out[gi].y, d_out[gi].z, d_out[gi].w, gi);
		d_out[gi].x += fixup;
        d_out[gi].y += fixup;
        d_out[gi].z += fixup;
        d_out[gi].w += fixup;
        // printf("After fixup: %d, %d, %d, %d, gi: %d\n", d_out[gi].x, d_out[gi].y, d_out[gi].z, d_out[gi].w, gi);
		gi += lws;
	}
}

__global__ void move_elements (int *d_in, int *d_positions, int* d_out, int nels){
    int gi = threadIdx.x + blockIdx.x * blockDim.x;
    if (gi >= nels) return;
    if (d_in[gi] % 2 == 0) return;
    
    int pos = d_positions[gi] - 1;

    d_out[pos] = d_in[gi];
}

void init_array(int *d_a, int n){
    int lws = 256;
    int ngroups = round_div_up(n, lws);
    init<<<ngroups, lws>>>(d_a);
}

void verify (int *h_a, int nels){
    for (int i = 0; i < nels/2; i++){
        if (h_a[i] != i*2+1)
            printf("Error at %d: %d != %d\n", i, h_a[i], i*2+1);
    }
}

int main (int argc, char * argv[]){
    if (argc != 4){
        printf("Usage: %s <nels> <lws> <ngroups>\n", argv[0]);
        exit(1);
    }
    int nels = atoi(argv[1]);
    int lws = atoi(argv[2]);
    int ngroups = atoi(argv[3]);

    int ntails = ngroups > 1 ? round_mul_up(ngroups, 4) : ngroups;

    int *h_a, *d_in, *d_positions, *d_out, *d_tails, *d_flags;
    float t1;
    cudaError_t err;

    size_t memsize = nels*sizeof(int);
    size_t memsize_host = nels/2*sizeof(int); // should be dynamically inferred from latest tail 
    size_t n_tails_size = ntails*sizeof(int);

    printf("nels: %d, lws: %d, ngroups: %d\n", nels, lws, ngroups);
    printf("ntails: %d, nquarts: %d\n", ntails, nels/4);
    printf("nquarts_per_wg: %d\n", round_mul_up(round_div_up(nels/4, ngroups), 32));

    // event creation

    std::vector<std::string> events_names = { "pre_init", "post_init", "post_scan_partial", "post_scan_tails", "post_fixup", "post_memcpy", "post_verify" };
    std::map<std::string, cudaEvent_t> events = create_events(events_names);

    // memory allocation
    
    err = cudaMalloc    (&d_in, memsize);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMalloc    (&d_flags, memsize);
    cuda_err_check(err, __FILE__, __LINE__);
    
    err = cudaMalloc    (&d_positions, memsize);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMalloc    (&d_tails, n_tails_size);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMalloc    (&d_out, memsize_host);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMallocHost(&h_a, memsize_host);
    cuda_err_check(err, __FILE__, __LINE__);

    // init array

    err = cudaEventRecord(events["pre_init"]);
    cuda_err_check(err, __FILE__, __LINE__);

    init_array(d_in, nels);

    err = cudaEventRecord(events["post_init"]);
    cuda_err_check(err, __FILE__, __LINE__);

    // compute flags

    compute_predicate<<<ngroups, lws, 0>>>((int4 *)d_in, (int4 *)d_flags, round_div_up(nels, 4));

    // initial scan

    scan_sliding_window<<<ngroups, lws, lws*sizeof(int)>>>((int4*)d_flags, (int4*)d_positions, d_tails, round_div_up(nels, 4), 32);

    err = cudaEventRecord(events["post_scan_partial"]);
    cuda_err_check(err, __FILE__, __LINE__);

    // print tails
    // printf("tails: ");
    // for (int i = 0; i < ntails; i++){
    //     int tail;
    //     err = cudaMemcpy(&tail, d_tails+i, sizeof(int), cudaMemcpyDeviceToHost);
    //     cuda_err_check(err, __FILE__, __LINE__);
    //     printf("%u ", tail);
    // }
    // printf("\n");

    // scan tails

    if (ngroups > 1){
        scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_tails, (int4*)d_tails, NULL, round_div_up(ntails, 4), 32);
    }

    // print tails
    // printf("tails after second scan: ");
    // for (int i = 0; i < ntails; i++){
    //     int tail;
    //     err = cudaMemcpy(&tail, d_tails+i, sizeof(int), cudaMemcpyDeviceToHost);
    //     cuda_err_check(err, __FILE__, __LINE__);
    //     printf("%u ", tail);
    // }
    // printf("\n");

    err = cudaEventRecord(events["post_scan_tails"]);
    cuda_err_check(err, __FILE__, __LINE__);
    
    // fixup tails

    if (ngroups > 1){
        scan_fixup<<<ngroups, lws>>>((int4*)d_positions, d_tails, round_div_up(nels, 4), 32);
    }

    // for (int i = 0; i < nels; i++){
    //     int pos;
    //     err = cudaMemcpy(&pos, d_positions+i, sizeof(int), cudaMemcpyDeviceToHost);
    //     cuda_err_check(err, __FILE__, __LINE__);
    //     printf("%u ", pos);
    // }
    // printf("\n");

    err = cudaEventRecord(events["post_fixup"]);
    cuda_err_check(err, __FILE__, __LINE__);
    
    // move elements

    printf("number of elements to move: %d\n", nels);
    move_elements<<<ngroups * 4, lws>>>(d_in, d_positions, d_out, nels);

    // copy back

    err = cudaMemcpy(h_a, d_out, memsize_host, cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventRecord(events["post_memcpy"]);
    cuda_err_check(err, __FILE__, __LINE__);

    // verify

    verify(h_a, nels);

    err = cudaEventRecord(events["post_verify"]);
    cuda_err_check(err, __FILE__, __LINE__);

    // print timings

    cudaEventElapsedTime(&t1, events["pre_init"], events["post_init"]);
    printf("init : %f ms (%f GB/s)\n", t1, memsize/t1/1e6);

    cudaEventElapsedTime(&t1, events["post_init"], events["post_scan_parrtial"]);
    printf("scan[partial]: %gms, %gGB/s\n", t1, (2*memsize+(ngroups > 1 ? n_tails_size : 0))/t1/1.0e6);

    if (ngroups > 1) {
        cudaEventElapsedTime(&t1, events["post_scan_partial"], events["post_scan_tails"]);
        printf("scan[tails]: %gms, %gGB/s\n", t1, n_tails_size/t1/1.0e6);

        cudaEventElapsedTime(&t1, events["post_scan_tails"], events["post_fixup"]);
        printf("fixup: %gms, %gGB/s\n", t1, (2*(nels - lws) + ngroups)*sizeof(int)/t1/1.0e6);
    }

    cudaEventElapsedTime(&t1, events["post_fixup"], events["post_memcpy"]);
    printf("copy : %f ms (%f GB/s)\n", t1, memsize/t1/1e6);

    cudaEventElapsedTime(&t1, events["post_memcpy"], events["post_verify"]);
    printf("verify: %f ms (%f GB/s)\n", t1, memsize/t1/1e6);

    // free memory

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tails);
    cudaFreeHost(h_a);
    for (auto event : events)
        cudaEventDestroy(event.second);
}