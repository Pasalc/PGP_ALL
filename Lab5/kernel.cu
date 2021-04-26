
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
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

#define assert(val) 

const int gridSize = 64;
const int blockSize = 64;
const int countNum = 256;
const int radixNum = 3;
const int bitsShift = 8;
const int countGrid = 1024;
//constexpr int DataSize = 100000000;

template<class T>
void printDev(T * data, int size) {
    int* h = (int*)malloc(sizeof(T) * size);
    CSC(cudaMemcpy(h, data, size * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        std::cout << "i:" << i << " " << h[i] << "\n";
    }
    free(h);
}
template<class T>
void printDev(T* data, int size, int shift) {
    T* h = (T*)malloc(sizeof(T) * size);
    CSC(cudaMemcpy(h, data, size * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        std::cout << "i:" << i << " " << (((h[i])>>shift )& 0xff) << "\n";
    }
    free(h);
}
template<class T>
int countDev(T* data, int size) {
    int* h = (int*)malloc(sizeof(T) * size);
    CSC(cudaMemcpy(h, data, size * sizeof(T), cudaMemcpyDeviceToHost));
    int c=0;
    for (int i = 0; i < size; ++i) {
        c+=h[i];
    }
    std::cout << "c: "<<c<<"\n";
    free(h);
    return c;

}
template<class T>
int countDev(T* data, int size,int size2,int off) {
    int* h = (int*)malloc(sizeof(T) * size*off);
    CSC(cudaMemcpy(h, data, size * sizeof(T) * off, cudaMemcpyDeviceToHost));
    int c = 0;
    int i = 0;
    int count = 0;
    while (i < size) {
        for (int j = 0; j < size2; j++) {
            c += h[count*off+j];
        }
        i++;
        ++count;
    }
    std::cout << "c: " << c << "\tcount: "<<count<<"\n";
    free(h);
    return c;

}

template<int bSize>
__device__ void tail_unroll(volatile int* buf, int tid) {
    if (bSize >= 64) {if(tid<32) buf[tid] += buf[tid + 32]; }
    if (bSize >= 32) {if(tid<16) buf[tid] += buf[tid + 16]; }
    if (bSize >= 16) {if(tid<8) buf[tid] += buf[tid + 8]; }
    if (bSize >= 8) {if(tid<4) buf[tid] += buf[tid + 4]; }
    if (bSize >= 4) {if(tid<2) buf[tid] += buf[tid + 2]; }
    if (bSize >= 2) {if(tid<1) buf[tid] += buf[tid + 1]; }
}
//grid=countNum
//block=gridSize/2
template<int blockSizeT>
__global__ void reduce_hist(int* out, int* hist) {
    __shared__ int buf[blockSizeT];
    const int tid = threadIdx.x;
	assert(countNum == 256);
    assert(blockSizeT == gridSize / 2);
    assert(gridDim.x == countNum);
    for (int c = 0; c < countGrid; ++c) {
        int offset = c * 2 * blockSizeT * countNum;//c * gridSize * countNum;
		assert(tid + blockSizeT + 2 * blockDim.x * blockIdx.x + offset<(c+1)*countNum*gridSize);
        buf[tid] = hist[tid + 2 * blockDim.x * blockIdx.x + offset] + hist[tid + blockSizeT + 2 * blockDim.x * blockIdx.x + offset];
        __syncthreads();
        for (int temp = blockSizeT / 2; temp > 64; temp /= 2) {
            if (tid < temp) {
				assert(tid+temp<blockSizeT);
                buf[tid] += buf[tid + temp];
            }
            __syncthreads();
        }

        __syncthreads();
        if (tid < 32) { tail_unroll<blockSizeT>(buf, tid); }

        __syncthreads();
        int offset_out = c * countNum;
        if (tid == 0) {
			assert(blockIdx.x+offset_out<(c+1)*countNum);
			assert(blockIdx.x+offset_out<countGrid*countNum);
            //0 : |012..255| 1: |012..255| ... countGrid:|012..255|
            //len(|........|)=countNum=256
            out[blockIdx.x+offset_out] = buf[0];
        }
    }
}
__global__ void histogramD(int* out, int* data, int size, int radix)
{
    __shared__ int count[countNum];
    int tid = threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    int bSize = size / countGrid + 1;
    int shift = radix * bitsShift;
    int mask = 0xFF << shift;
    for (int i = tid; i < countNum; i += blockDim.x) {
        count[i] = 0;
    }
    __syncthreads();
    for (int c = 0; c < countGrid; ++c) {
        for (int i = tid + blockDim.x * blockIdx.x + bSize * c; i < bSize * (c + 1) && i<size; i += gridSize) {
            assert(i < size);
            int val = (data[i] & mask) >> shift;
            assert(val < countNum);
            atomicAdd(&count[val], 1);
        }
        __syncthreads();
        for (int i = tid; i < countNum; i += blockDim.x) {

            assert(count[i] >= 0);
            //0 : |0..0|1..1|...|255..255| 1:|0..0|... countGrid: |0..0|...|255..255| 
            //len(|....|)=gridDim.x=gridSize
            assert(i * gridDim.x + blockIdx.x + gridSize * countNum * c < gridSize * countNum * countGrid);
            out[i * gridDim.x + blockIdx.x + gridDim.x * countNum * c] = count[i];
        }
        __syncthreads();
    }
}

void histogram(int* d_histogram, int* d_data, int size, int radix) {
    int* d_hist_out;
    CSC(cudaMalloc(&d_hist_out, sizeof(int) * gridSize * countNum * countGrid));//CSC(cudaMalloc(&d_hist_out, sizeof(int) * radixNum * gridSize * countNum *countGrid));
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());
    histogramD<<<gridSize, blockSize >>> (d_hist_out, d_data, size, radix);
    CSC(cudaDeviceSynchronize());
    reduce_hist<gridSize / 2> << <countNum, gridSize / 2 >> > (d_histogram , d_hist_out);
    CSC(cudaDeviceSynchronize());
    CSC(cudaFree(d_hist_out));
}

__global__ void rSort(int* out, int* hist, int* data, int size, int radix) {
    __shared__ int count[countNum];
    assert(blockDim.x == 1);
    int shift = radix * bitsShift;
    int mask = 0xFF << shift;
    int bSize = size / countGrid + 1;
    int bid = blockIdx.x;
    for (int i = 0, off_hist = countNum * blockIdx.x; i < countNum; i += blockDim.x) {
        count[i] = hist[i+off_hist];
    }
    //__syncthreads();
    for (int i = bSize*bid; i< bSize * (bid+1) && i < size; i += blockDim.x) {
        int val = data[i];
        int pos = (count[(val & mask) >> shift])++;
        out[pos] = val;
    }
}
__global__ void rCheck(int* data, int size, int radix) {

    int bSize = size / countGrid + 1;
    int bid = blockIdx.x;
    int shift = radix * bitsShift;
    for (int i = bSize * bid; i < bSize * (bid + 1) && i < size-1; i += blockDim.x) {
        if((data[i] >> shift) & 0xff > (data[i + 1] >> shift) & 0xff)
            assert(((data[i]>>shift)&0xff) <= ((data[i+1] >> shift) & 0xff));
    }
}
void radSort(int* d_hist, int* d_data,int size,int radix) {
    int* temp;
    CSC(cudaMalloc(&temp,sizeof(int) * size));
    rSort << <countGrid, 1>> > (temp,d_hist,d_data,size,radix);
    CSC(cudaDeviceSynchronize());
    //rCheck << <countGrid, 1 >> > (temp, size, radix);
    CSC(cudaDeviceSynchronize()); 
    CSC(cudaMemcpy(d_data, temp, sizeof(int) * size, cudaMemcpyDeviceToHost));
    CSC(cudaFree(temp));

}
//begin position of i number

__global__ void hist_begin(int* out, int* hist, int* count) {
    int tid = threadIdx.x;
    int off2 = countNum * blockIdx.x;
    int i = tid + off2;
    int val = hist[i] + count[tid];
    assert(i>= 0);
    assert(i < countNum * countGrid);
    assert(tid< countNum);
    out[i] = val;
}
__global__ void hist_check(int* hist, int cSize) {
    int tid = threadIdx.x;
    for (int i = tid + blockDim.x * blockIdx.x; i < cSize - 1; i += blockDim.x * gridDim.x) {
        if (i % countNum != countNum - 1 && hist[i] > hist[i + 1])
            assert(hist[i] <= hist[i + 1]);
        if (i % countGrid > countNum&& hist[i - countNum] > hist[i])
            assert(hist[i - countNum ] <= hist[i]);
    }
}
__global__ void count_check(int* hist, int cSize, int size) {
    int tid = threadIdx.x;
    for (int i = tid + blockDim.x * blockIdx.x; i < cSize - 1; i += blockDim.x * gridDim.x) {
        if (i % countNum != countNum - 1 && hist[i] > hist[i + 1])
            assert(hist[i] <= hist[i + 1]);
    }
    if(hist[cSize - 1] != size)
        assert(hist[cSize - 1] == size);
}
void hist_radix(int* hist) {
    int* d_hist_ext, * d_count;
    CSC(cudaMalloc(&d_hist_ext, sizeof(int) * countNum * (countGrid+1)));
    CSC(cudaMemset(d_hist_ext, 0, sizeof(int) * countNum));
    CSC(cudaMemcpy(d_hist_ext + countNum, hist, sizeof(int) * countNum * countGrid, cudaMemcpyDeviceToDevice));
    
    CSC(cudaMalloc(&d_count, sizeof(int) * (countNum+1)));
    CSC(cudaMemset(d_count, 0, sizeof(int) * (countNum + 1)));
    CSC(cudaMemcpy(d_count + 1, hist + countNum * (countGrid - 1), sizeof(int) * (countNum-1), cudaMemcpyDeviceToDevice));
    
    hist_begin << <countGrid, countNum>> > (hist, d_hist_ext,d_count);
    CSC(cudaDeviceSynchronize());
    CSC(cudaFree(d_hist_ext));
    CSC(cudaFree(d_count));
}

void radixSort(int* h_out, int* d_data, int size) {

    device_vector<int> t_histogram(countNum * countGrid, 0);
    int* d_histogram = raw_pointer_cast(t_histogram.data());
    for (int radix = 0; radix < radixNum; ++radix) {
        std::cerr << "r:" << radix << "\n";
        histogram(d_histogram, d_data, size,radix);
        CSC(cudaDeviceSynchronize());
        int off = (countGrid - 1) * countNum;
        inclusive_scan(t_histogram.begin() + off, t_histogram.begin() + countNum + off, t_histogram.begin() + off);
        CSC(cudaDeviceSynchronize());
        //count_check << <countGrid, countNum >> > (d_histogram + off, countNum, size);
        CSC(cudaDeviceSynchronize());
        hist_radix(d_histogram);
        CSC(cudaDeviceSynchronize());
        radSort(d_histogram, d_data, size,radix);
        CSC(cudaDeviceSynchronize());
    }
    CSC(cudaMemcpy(h_out, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
    //std::cerr << "countSort out\n";
}

int main()
{
    int size;
    freopen(NULL, "rb", stdin);
    fread(&size, sizeof(size), 1, stdin);
    std::cerr << size << '\n';
	std::cerr << gridSize<<"gb"<<blockSize << '\n';
	
    int* h_data=(int*)malloc(sizeof(int)*size);
    if (h_data == NULL) {
        std::cerr << "Not enough mem\n";
        exit(0);
    }
    //assert(h_data != NULL);
    fread(h_data, sizeof(int), size, stdin);
    for (int i = 0; i < size; ++i) {
		//h_data[i]=i^(i-1)%(1<<24);
        //std::cerr << h_data[i]<<" ";
    }
    std::cerr << '\n';
    int* d_data;
    CSC(cudaMalloc(&d_data, sizeof(int) * size));
    CSC(cudaMemcpy(d_data, h_data, sizeof(int) * size, cudaMemcpyHostToDevice));
    std::cerr << "Radix sort begin\n";

	radixSort(h_data, d_data, size);
    std::cerr << "Radix sort end\n";

    freopen(NULL, "wb", stdout);
    fwrite(h_data, sizeof(int), size, stdout);
    std::cerr << "Write End\n";

    free(h_data);
    std::cerr << "Free End\n";

    /*
    for (int i = 0; i < 20 && i < size; ++i) {
        std::cerr << h_data[i] << " ";
    }
    */
    CSC(cudaFree(d_data));
    std::cerr << "cudaFree End\n";

    fclose(stdin);
    std::cerr << "???";
    fclose(stdout);
    std::cerr << "???\n";
	std::cerr <<"Time:"<<t<<std::endl;
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CSC(cudaDeviceReset());
    return 0;
}
