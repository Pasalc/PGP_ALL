
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <random>

using namespace thrust;

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)
	
#define START_RECORD            \
CSC(cudaEventCreate(&start));   \
CSC(cudaEventCreate(&end));     \
CSC(cudaEventRecord(start));    
#define END_RECORD                          \
CSC(cudaEventRecord(end));                  \
CSC(cudaEventSynchronize(end));             \
CSC(cudaEventElapsedTime(&t, start, end));  \
CSC(cudaEventDestroy(start));               \
CSC(cudaEventDestroy(end));

template<class T>
void printDev(T * data, int size) {
    int* h = (int*)malloc(sizeof(T) * size);
    CSC(cudaMemcpy(h, data, size * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        std::cerr << "i:" << i << " " << h[i] << "\n";
    }
    free(h);
}

constexpr int blockSize = 32;
constexpr int gridSize = 32;

template<int blockSize>
__global__ void kernel(double* out, double* mat,const int height, const int width)
{
    assert(blockDim.x == blockDim.y);
    assert(blockDim.x==blockSize);
    __shared__ double buf[blockSize][blockSize+1];
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    //int gridSize = gridDim.x * blockDim.x;
    int gsx = blockDim.x * gridDim.x;
    int gsy = blockDim.y * gridDim.y;
    for (int i = tidy + blockDim.y * blockIdx.y; i < height; i += gsy) {
        for (int j = tidx + blockDim.x * blockIdx.x; j < width; j += gsx) {
            buf[tidx][tidy] = mat[i * width + j];
            __syncthreads();
            int ys = min(height - i + tidy, blockSize);
            int xs = min(width - j + tidx, blockSize);
            assert(tidx < xs);
            int posx = (tidx + tidy * xs) % ys;
            int posy = (tidx + tidy * xs) / ys;
            int offx = i - tidy;
            int offy = j - tidx;
            out[(offy+posy) * height + posx+offx] = buf[posy][posx];
            __syncthreads();

        }
    }
}
__global__ void kernel2(double* tran, double* mat, int height, int width)
{
    int tid = threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;
    for (int id = tid + blockIdx.x * blockDim.x; id < width * height; id += gridSize) {
        int idTran = id / width + (id % width) * height;
        tran[idTran] = mat[id];
    }
}

int main()
{
    srand(4);
    std::cin.tie(NULL);
    std::ios_base::sync_with_stdio(false);
    int n=64, m=64;
    std::cin >> n >> m;
    //n = m = 1024;
    //n = 3000; m = 1000;
    host_vector<double> h_mat(n * m);
    for (int i = 0; i < n * m; ++i) {
        //std::cin >> h_mat[i];
        h_mat[i] = i;
    }
    std::cerr << "n:" << n << " m:" << m << "\n";
    std::cerr << "Mat:\n";
    for (int i = 0; i < 8 && i<n; ++i) {
        for (int j = 0; j < 8 && j<m; ++j)
            std::cerr << h_mat[i * m + j] << " ";
        std::cerr << "\n";
    }
    device_vector<double> t_mat = h_mat;
    device_vector<double> t_tran(n * m);
    double* d_mat = thrust::raw_pointer_cast(t_mat.data());
    double* d_tran = thrust::raw_pointer_cast(t_tran.data());
	float t;
    cudaEvent_t start, end;
    CSC(cudaGetLastError());
    START_RECORD
    kernel<blockSize> << <dim3(gridSize, gridSize), dim3(blockSize, blockSize) >> > (d_tran, d_mat, n, m);
	END_RECORD
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());
    h_mat = t_tran;
    std::cerr << "Time:"<<t<<"\n";
    std::cerr << "TMat:\n";
    for (int i = 0; i < 8 && i < n; ++i) {
        for (int j = 0; j < 8 && j < m; ++j)
            std::cerr << h_mat[i * n + j] << " ";
        std::cerr << "\n";
    }
    for (int i = 0; i < n * m; ++i) {
        printf("%lf", h_mat[i]);
        if (i % n == n - 1) {
            printf("\n");
        }
        else {
            printf(" ");
        }
    }
    return 0;
}
