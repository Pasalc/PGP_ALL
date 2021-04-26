#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <corecrt_malloc.h>
#include <ios>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include <assert.h>

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)
#define SET_RGB(byte,alpha)   ((alpha)<<24 | (byte)<<16 | (byte)<<8 | (byte))
#define SET_BW(p) ((double)(0.299*p.x+0.587*p.y+0.114*p.z))
texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void conturSobel(uint32_t* out, int w, int h)
{
    //__shared__ image[16][16]
    const int tx = blockIdx.x*blockDim.x + threadIdx.x;
    const int ty = blockIdx.y*blockDim.y + threadIdx.y;
    const int offx = blockDim.x * gridDim.x;
	const int offy = blockDim.y * gridDim.y;
    int x,y;
    uchar4 p;
    int temp;
    double sum;
    double sum2;
    for(y=ty;y<h;y+=offy){
        for(x=tx;x<w;x+=offx){
            sum=0;
            sum2=0;
            //Gx
            p = tex2D(tex, x-1, y-1);
			sum -= SET_BW(p);
            p = tex2D(tex, x-1, y);
			sum -= 2*SET_BW(p);
            p = tex2D(tex, x-1, y+1);
			sum -= SET_BW(p);
            
            p = tex2D(tex, x+1, y-1);
			sum += SET_BW(p);
            p = tex2D(tex, x+1, y);
			sum += 2*SET_BW(p);
            p = tex2D(tex, x+1, y+1);
			sum += SET_BW(p);
            //Gy
            p = tex2D(tex, x-1, y-1);
			sum2 -= SET_BW(p);
            p = tex2D(tex, x, y-1);
			sum2 -= 2*SET_BW(p);
            p = tex2D(tex, x+1, y-1);
			sum2 -= SET_BW(p);
            
            p = tex2D(tex, x-1, y+1);
			sum2 += SET_BW(p);
            p = tex2D(tex, x, y+1);
			sum2 += 2*SET_BW(p);
            p = tex2D(tex, x+1, y+1);
			sum2 += SET_BW(p);
            //f
            temp=min((int)((sqrt(sum*sum+sum2*sum2))),255);
            out[y * w + x]=SET_RGB(temp,tex2D(tex, x, y).w);
        }
    }   
}
int main()
{
    std::string str;
    FILE *in, *out;
    assert(sizeof(int)==4);
    std::getline(std::cin,str);
    in=fopen(str.c_str(), "rb");
    std::getline(std::cin, str);
    out=fopen(str.c_str(), "wb");
    uint32_t w, h, *h_image,*d_image;
    fread(&w, sizeof(uint32_t), 1, in);
    fread(&h, sizeof(uint32_t), 1, in);
    h_image=(uint32_t*)malloc(sizeof(uint32_t)*w*h);
    fread(h_image,sizeof(uint32_t),w*h,in);
    
    cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
	CSC(cudaMemcpyToArray(arr, 0, 0, h_image, 4 * w * h, cudaMemcpyHostToDevice));

    // Подготовка текстурной ссылки, настройка интерфейса работы с данными
	tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	tex.normalized = false;						// Режим нормализации координат: без нормализации
    // Связываем интерфейс с данными
	CSC(cudaBindTextureToArray(tex, arr, ch));

    CSC(cudaMalloc(&d_image,4*w*h));
    cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));

	conturSobel<<<dim3(16,16),dim3(16,16)>>>(d_image,w,h);
	CSC(cudaGetLastError());
    CSC(cudaMemcpy(h_image,d_image,4*w*h,cudaMemcpyDeviceToHost));

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));
    CSC(cudaUnbindTexture(tex));
	CSC(cudaFreeArray(arr));
    CSC(cudaFree(d_image));
    fwrite(&w, 4, 1, out);
    fwrite(&h, 4, 1, out);
    fwrite(h_image,4,w*h,out);
    
    free(h_image);
    fclose(in);fclose(out);
    return 0;
}
