#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)


__global__ void min_vec(double* arr, double* arr2, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while (idx < n) {
		arr[idx] = arr[idx]<arr2[idx] ? arr[idx] : arr2[idx];
		idx += offset;
	}
}

int main() {
	int n;
	std::cin >> n;
	double* h_arr	= (double*)malloc(sizeof(double) * n);
	double* h_arr2	= (double*)malloc(sizeof(double) * n);

	for (double* arr_p = h_arr; arr_p < h_arr+n; ++arr_p) {
		std::cin >> (*arr_p);
	}
	for (double* arr_p = h_arr2; arr_p < h_arr2+n; ++arr_p) {
		std::cin >> (*arr_p);
	}

	double* d_arr;
	double* d_arr2;

	CSC(cudaMalloc(&d_arr, sizeof(double) * n));
	CSC(cudaMemcpy(d_arr, h_arr, sizeof(double) * n, cudaMemcpyHostToDevice));
	CSC(cudaMalloc(&d_arr2, sizeof(double) * n));
	CSC(cudaMemcpy(d_arr2, h_arr2, sizeof(double) * n, cudaMemcpyHostToDevice));

	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));

	min_vec << <256, 256 >> > (d_arr, d_arr2, n);
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));

	//printf("time = %f\n", t);

	CSC(cudaMemcpy(h_arr, d_arr, sizeof(double) * n, cudaMemcpyDeviceToHost));
	CSC(cudaFree(d_arr));
	CSC(cudaFree(d_arr2));

	for (int i = 0; i < n; ++i)
		printf("%f ", h_arr[i]);
	printf("\n");
	free(h_arr);
	free(h_arr2);
	return 0;
}
