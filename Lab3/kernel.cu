
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory>
#include <iostream>
#include <ios>
#include <string>
#include <fstream>
#include <string>
#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)
/*
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
*/
#define BLOCK_SIZE 64
__constant__ uint32_t var[1];

__global__ void kernel(uchar4* out,int size){
    int tid = threadIdx.x;
    int gridSize = gridDim.x*blockDim.x;
    double3 p;
    double res,res2,res3;
    int arg = 0;
    for(int i = tid + blockDim.x * blockIdx.x; i < size; i += gridSize) {
        p.x = out[i].x;
        p.y = out[i].y;
        p.z = out[i].z;
        res = -((var[0] - p.x) * (var[0] - p.x) + p.y * p.y + p.z * p.z);
        res2 = -(p.x * p.x + (var[0] - p.y) * (var[0] - p.y) + p.z * p.z);
        res3 = -(p.x * p.x + p.y * p.y + (var[0] -p.z) * (var[0] -p.z));
        arg = res >= res2 ? (res >= res3 ? 0 : 2) : (res2 >= res3 ? 1 : 2);
        out[i].w = arg;
    }
}
int main()
{
    constexpr int blockSize = BLOCK_SIZE;
    constexpr int gridSize = 64;
    std::string str;
    FILE* in, * out;
    uchar4 *d_image;
    
    //DataRead
    std::getline(std::cin, str);
    in = fopen(str.c_str(), "rb");
    std::getline(std::cin, str);
    out = fopen(str.c_str(), "wb");
    if (in == NULL) {
        fprintf(stderr, "ERROR in %s:%d: Can't open input file\n", __FILE__, __LINE__);
        exit(0);
    }
    if (out == NULL) {
        fprintf(stderr, "ERROR in %s:%d: Can't open output file\n", __FILE__, __LINE__);
        exit(0);
    }
    uint32_t width, height, * h_image;
	if (fread(&width, sizeof(uint32_t), 1, in) != 1) {
        fprintf(stderr, "ERROR in %s:%d: Can't read width\n", __FILE__, __LINE__);
        exit(0);
    }
    if (fread(&height, sizeof(uint32_t), 1, in) != 1) {
        fprintf(stderr, "ERROR in %s:%d: Can't read height\n", __FILE__, __LINE__);
        exit(0);
    }
    if (width == 0 || height==0) {
        fprintf(stderr, "ERROR in %s:%d: zero size image: w:%d h:%d\n", __FILE__, __LINE__,width,height);
        fwrite(&width, 4, 1, out);
        fwrite(&height, 4, 1, out);
        fclose(in); fclose(out);
        CSC(cudaDeviceReset());
        exit(0);
    }
    std::cerr<<width<<"wh"<<height<<"\n";
    h_image = (uint32_t*)malloc(sizeof(uint32_t) * width * height);									
    if (h_image == nullptr) {																		
        fprintf(stderr, "ERROR in %s:%d:can't allocate memory", __FILE__, __LINE__);				
        exit(0);																					
    }
    h_image[0] = 255;
    CSC(cudaMemcpyToSymbol(var, h_image, sizeof(uint32_t)));
    fread(h_image, sizeof(uint32_t), width * height, in);
    													                                  
    CSC(cudaMalloc(&d_image, sizeof(uchar4) * width * height));										
    CSC(cudaMemcpy(d_image, h_image, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice));

    /*
    float t;
    cudaEvent_t start, end;
    CSC(cudaGetLastError());
    START_RECORD
    */

    kernel << <gridSize, blockSize >> > (d_image,width*height);
    //END_RECORD
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(h_image, d_image, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
    
    //Out
    fwrite(&width, 4, 1, out);
    fwrite(&height, 4, 1, out);
    fwrite(h_image, sizeof(uint32_t), width * height, out);
    //Clear
    CSC(cudaFree(d_image));
    free(h_image);
    fclose(in); fclose(out);
    CSC(cudaDeviceReset());
    return 0;
}