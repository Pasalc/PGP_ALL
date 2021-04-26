
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string>
#include <limits>
#include <unistd.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));	\
		MPI_Finalize();				\
		exit(0);					\
	}								\
} while(0)

#define INDEX(i,j,linesize) (i)*(linesize)+(j)
#define INDEX_LEFT_OUT(i,linesize) (i+1)*(linesize)+1
#define INDEX_RIGHT_OUT(i,linesize) (i+2)*(linesize)-2
#define INDEX_UP_OUT(j,linesize) (linesize)+1+j
#define INDEX_DOWN_OUT(j,linesize,bys) (bys)*(linesize)+1+j

#define INDEX_LEFT_IN(i,linesize) (i+1)*(linesize)
#define INDEX_RIGHT_IN(i,linesize) (i+2)*(linesize)-1
#define INDEX_UP_IN(j,linesize) 1+j
#define INDEX_DOWN_IN(j,linesize,bys) ((bys)+1)*(linesize)+1+j

using tvec=thrust::device_vector<double>;

#define LEFT_KEY 0
#define RIGHT_KEY 1
#define UP_KEY 2
#define DOWN_KEY 3

#define blockSize 32

template<class T>
void printDev(T * data, int size) {
    T* h = (T*)malloc(sizeof(T) * size);
    CSC(cudaMemcpy(h, data, size * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        std::cout << "i:" << i << " " << h[i] << "\n";
    }
    free(h);
}
double Calc_local_eps(double* u,double* uprev, int bxs,int bys){
	int i;
	tvec tleps(bxs*bys,0);
	for(i = 1;i<bys+1;++i){
		thrust::transform(thrust::device,u+(bxs+2)*i+1, u+(bxs+2)*(i+1)-1, uprev+(bxs+2)*i+1, tleps.begin()+(i-1)*bxs,
			thrust::minus<double>());
	}
	auto maxe = thrust::minmax_element(tleps.begin(), tleps.end());
	return max(abs(*(maxe.first)),abs(*(maxe.second)));
}

__global__ void UIter(double* unext, double* u,double hdx, double hdy,int bxs, int bys){//hdy=1/(hds**2)//REM
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int gridSizeX=gridDim.x*blockDim.x;
	int gridSizeY=gridDim.y*blockDim.y;
	int ls=bxs+2;
	for(int i=1+ty+by*blockDim.y;i<bys+1;i+=gridSizeY){
		for(int j=1+tx+bx*blockDim.x;j<bxs+1;j+=gridSizeX){
			unext[INDEX(i,j,ls)]=((u[INDEX(i+1,j,ls)]+u[INDEX(i-1,j,ls)])*hdx+(u[INDEX(i,j+1,ls)]+u[INDEX(i,j-1,ls)])*hdy)/(2*(hdx+hdy));
		}
	}
}

__global__ void borderOut(double* out,double* u, int bxs,int bys){
	int tx=threadIdx.x;
	int bdx=blockDim.x;
	int gx=gridDim.x;
	int gridSize=bdx*gx;
	int ls=bxs+2;
	int offl=0;
	int offr=offl+bys;
	int offu=offr+bys;
	int offd=offu+bxs;
	//Left
	for(int i=tx+blockIdx.x*blockDim.x;i<bys;i+=gridSize){
		out[offl+i]=u[INDEX_LEFT_OUT(i,ls)];
	}
	//Right
	for(int i=tx+blockIdx.x*blockDim.x;i<bys;i+=gridSize){
		out[offr+i]=u[INDEX_RIGHT_OUT(i,ls)];
	}
	//Up
	for(int j=tx+blockIdx.x*blockDim.x;j<bxs;j+=gridSize){
		out[offu+j]=u[INDEX_UP_OUT(j,ls)];
	}
	//Down
	for(int j=tx+blockIdx.x*blockDim.x;j<bxs;j+=gridSize){
		out[offd+j]=u[INDEX_DOWN_OUT(j,ls,bys)];
	}
}
__global__ void borderIn(double* out, double* border, int bxs, int bys, int left, int right, int up, int down){
	int tx=threadIdx.x;
	int bdx=blockDim.x;
	int gx=gridDim.x;
	int gridSize=bdx*gx;
	int ls=bxs+2;
	int offl=0;
	int offr=offl+bys;
	int offu=offr+bys;
	int offd=offu+bxs;
	//Left
	if(left!=-1){
		for(int i=tx+blockIdx.x*blockDim.x;i<bys;i+=gridSize){
			out[INDEX_LEFT_IN(i,ls)]=border[i+offl];
		}
	}
	//Right
	if(right!=-1){
		for(int i=tx+blockIdx.x*blockDim.x;i<bys;i+=gridSize){
			out[INDEX_RIGHT_IN(i,ls)]=border[i+offr];
		}
	}
	//Up
	if(up!=-1){
		for(int j=tx+blockIdx.x*blockDim.x;j<bxs;j+=gridSize){
			out[INDEX_UP_IN(j,ls)]=border[j+offu];
		}
	}
	//Down
	if(down!=-1){
		for(int j=tx+blockIdx.x*blockDim.x;j<bxs;j+=gridSize){
			out[INDEX_DOWN_IN(j,ls,bys)]=border[j+offd];
		}
	}
} 
__global__ void init_matr(double* out, double uzero,int bxs, int bys){
	int tx=threadIdx.x;
	int bdx=blockDim.x;
	int gx=gridDim.x;
	int gridSize=bdx*gx;
	for(int i=tx+blockIdx.x*blockDim.x;i<(bys+2)*(bxs+2);i+=gridSize){
		out[i]=uzero;
	}
	
}

__global__ void init_border(double* out, int bxs, int bys, double ul, double ur, double uu, double ud, int left, int right, int up, int down){
	int tx=threadIdx.x;
	int bdx=blockDim.x;
	int gx=gridDim.x;
	int gridSize=bdx*gx;
	int ls=bxs+2;
	/*
	int offl=0;
	int offr=offl+bys;
	int offu=offr+bys;
	int offd=offu+bxs;
	*/
	if(down==-1){
		for(int j=tx+blockIdx.x*blockDim.x;j<bxs;j+=gridSize){
			out[INDEX_DOWN_IN(j,ls,bys)]=ud;
		}
	}
	if(up==-1){
		for(int j=tx+blockIdx.x*blockDim.x;j<bxs;j+=gridSize){
			out[INDEX_UP_IN(j,ls)]=uu;
		}
	}
	if(left==-1){
		for(int i=tx+blockIdx.x*blockDim.x;i<bys;i+=gridSize){
			out[INDEX_LEFT_IN(i,ls)]=ul;
		}
	}
	if(right==-1){
		for(int i=tx+blockIdx.x*blockDim.x;i<bys;i+=gridSize){
			out[INDEX_RIGHT_IN(i,ls)]=ur;
		}
	}
}

int main(int argc, char* argv[]) {
	int pxs,pys;
	int bxs,bys;
	std::string out_str("out.txt");
	int str_len;
	double mineps,leps;
	double lx,ly;
	double ul,ur,uu, ud;
	double uzero;
	
	MPI_Init(&argc, &argv);
	int i,j, wrank, wnprocs; 
	MPI_Comm_size(MPI_COMM_WORLD, &wnprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
	if(wrank==0){
		std::cin>>pxs>>pys;
		std::cin>>bxs>>bys;
		std::cin>>out_str;
		std::cin>>mineps;
		std::cin>>lx>>ly;
		std::cin>>ul>>ur>>uu>>ud;
		std::cin>>uzero;
		str_len=out_str.length();
	}
	MPI_Bcast(&pxs,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&pys,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&bxs,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&bys,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&str_len,1,MPI_INT,0,MPI_COMM_WORLD);
	out_str.resize(str_len);
	MPI_Bcast((void*)out_str.data(),str_len,MPI_CHAR,0,MPI_COMM_WORLD);
	MPI_Bcast(&mineps,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	MPI_Bcast(&lx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ly,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	MPI_Bcast(&ul,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ur,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&uu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ud,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&uzero,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	if(wrank==1){
		fprintf(stderr,"A %d (%d):pxs:%d pys:%d bxs:%d bys:%d lx:%lf ly:%lf ul:%lf ur:%lf uu:%lf ud:%lf uzero:%lf  mineps:%lf %s\n",wrank, wnprocs , pxs,pys,bxs,bys,lx,ly,ul,ur,uu,ud,uzero,mineps, out_str.c_str());
	}
	
	int linesize=bxs+2;
	int n=linesize*(bys+2);
	double hxs=lx/(bxs*pxs);
	double hys=ly/(bys*pys);
	
	int color = wrank<pys*pxs ? 0: MPI_UNDEFINED;
	MPI_Comm my_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &my_comm);
	if(color==0){
		char *buf, *buf_p;
		int bsize;
		int rank, nprocs; 
		MPI_Comm_size(my_comm, &nprocs);
		MPI_Comm_rank(my_comm, &rank);
		int devicecount;
		CSC(cudaGetDeviceCount(&devicecount));
		CSC(cudaSetDevice(rank % devicecount));
		fprintf(stderr, "devR(%d): %d (%d)",rank,rank % devicecount,devicecount);
		MPI_Datatype vertvect;
		MPI_Type_vector(bys, 1, linesize, MPI_DOUBLE, &vertvect);
		MPI_Type_commit(&vertvect);
	//Init
		int x=rank%pxs,y=rank/pxs;
		int up,down,left,right;
		right = (x+1== pxs)? -1 : y*pxs + (x + 1) % pxs;
		left = (x-1==-1) ? -1 : y*pxs + (x - 1) % pxs;
		up= (y-1==-1) ? -1 :((y-1)*pxs) + x;
		down = (y+1==pys)? -1: ((y+1)*pxs) + x;
		//Init buf
		{
			int lrs, uds;
			MPI_Pack_size(bys,MPI_DOUBLE,my_comm,&lrs);
			MPI_Pack_size(bxs,MPI_DOUBLE,my_comm,&uds);
			int bufsize = 4*MPI_BSEND_OVERHEAD + 2*(lrs + uds);
			buf = (char *)malloc( bufsize );
			MPI_Buffer_attach(buf, bufsize);
		}
		if(rank==0)
			fprintf(stderr, "pxs %d,pys %d,bxs %d,bys %d,mineps %lf,lx %lf,ly %lf,ul %lf,ur %lf, uu %lf , ud %lf , uzero %lf, out_str: %s \n", pxs,pys,bxs,bys,mineps,lx,ly,ul,ur,uu,ud,uzero,out_str.c_str());
	//main
		int count=0;
		//tvec t_data(n,uzero), t_prev(n,uzero);
		double *d_data, *d_dataswap, *d_border;
		double *h_border_out, *h_border_in;
		//To free later
		h_border_in=(double*)malloc(sizeof(double)*(2*bxs+2*bys));
		h_border_out=(double*)malloc(sizeof(double)*(2*bxs+2*bys));
		CSC(cudaMalloc(&d_border,sizeof(double)*(2*bxs+2*bys)));
		CSC(cudaMalloc(&d_data,sizeof(double)*n));
		CSC(cudaMalloc(&d_dataswap,sizeof(double)*n));
		//Init margins
		init_matr<<<64,64>>>(d_data,uzero,bxs,bys);
		init_border<<<64,64>>>(d_data,bxs,bys,ul,ur,uu,ud,left,right,up,down);//borders only
		CSC(cudaGetLastError());
		CSC(cudaMemcpy(d_dataswap,d_data,sizeof(double)*n,cudaMemcpyDeviceToDevice));
		double hdx=1/(hys*hys);//Da, tak hdx ==h_x iz usloviya
		double hdy=1/(hxs*hxs);
		
		const int offl=0;
		const int offr=offl+bys;
		const int offu=offr+bys;
		const int offd=offu+bxs;
		fprintf(stderr, "%d %d %d %d\n",left, right, down, up);
		while(count<100000){
			MPI_Request req[4];
			MPI_Status statuses[4];
			//main
			//printDev(d_data,n);
			UIter<<<dim3(16,16),dim3(blockSize,blockSize)>>>(d_dataswap,d_data,hdx,hdy, bxs,bys);
			std::swap(d_data,d_dataswap);
			count++;
			//calc eps
			CSC(cudaDeviceSynchronize());
			//printDev(d_data,n);
			leps=Calc_local_eps(d_data,d_dataswap,bxs,bys);
			if(rank==0){
				//fprintf(stderr, "leps: %lf\n",leps);
			}
			MPI_Allreduce(MPI_IN_PLACE,&leps,1,MPI_DOUBLE,MPI_MAX,my_comm);
			if(rank==0){
				//fprintf(stderr, "leps: %lf\n",leps);
			}
			if(leps<mineps){
				free(h_border_in);
				free(h_border_out);
				CSC(cudaFree(d_border));
				break;
			}
			//!done
			//recv/send from up to down
			int req_count=0;
			//copydata from device
			borderOut<<<64,64>>>(d_border,d_data,bxs,bys);
			//printDev(d_border,(2*bxs+2*bys));
			CSC(cudaMemcpy(h_border_out,d_border,sizeof(double)*(2*bxs+2*bys),cudaMemcpyDeviceToHost));
			CSC(cudaMemcpy(h_border_in,d_border,sizeof(double)*(2*bxs+2*bys),cudaMemcpyDeviceToHost));
			//May Bsend all first, then Irecv all in same memory
			if(left!=-1){
				MPI_Bsend((void*)(h_border_out+offl), bys, MPI_DOUBLE, left, 0, my_comm);
				MPI_Irecv((void*)(h_border_in+offl), bys, MPI_DOUBLE, left, 0, my_comm, &req[req_count]);
				++req_count;
			}
			if(right!=-1){
				MPI_Bsend(h_border_out+offr, bys, MPI_DOUBLE, right, 0, my_comm);
				MPI_Irecv(h_border_in+offr, bys, MPI_DOUBLE, right, 0, my_comm, &req[req_count]);
				++req_count;
			}
			if(up!=-1){
			
				MPI_Bsend(h_border_out+offu, bxs, MPI_DOUBLE, up, 0, my_comm);
				MPI_Irecv(h_border_in+offu, bxs, MPI_DOUBLE, up, 0, my_comm,&req[req_count]);
				++req_count;
				}
			if(down!=-1){
				MPI_Bsend(h_border_out+offd, bxs, MPI_DOUBLE, down, 0, my_comm);
				MPI_Irecv(h_border_in+offd, bxs, MPI_DOUBLE, down, 0, my_comm,&req[req_count]);
				++req_count;
			}
			MPI_Waitall(req_count, req,statuses);
			CSC(cudaMemcpy(d_border,h_border_in, sizeof(double)*(2*bxs+2*bys),cudaMemcpyHostToDevice));//copy from host
			borderIn<<<64,64>>>(d_data,d_border,bxs,bys,left,right,up,down);
			CSC(cudaDeviceSynchronize());
		}
		MPI_Buffer_detach( &buf_p, &bsize );
	//Write
		//fprintf(stderr, "count : %d\n",count);
		double* data = (double*)malloc(sizeof(double) * n);
		CSC(cudaMemcpy(data,d_data,sizeof(double)*n,cudaMemcpyDeviceToHost));//copy from device
		int n_size = 15;
		char * buff_out = (char *) malloc(sizeof(char) * (bxs) * (bys)*n_size);
		memset(buff_out, ' ', (bxs) * (bys) * n_size * sizeof(char));
		
		for(j = 0; j < bys; ++j) {
			for(i = 0; i < bxs; ++i){
				sprintf(buff_out + (j * bxs + i)*n_size, " %.6e ", data[(j+1) * linesize + (i+1)]);
			}
			if (x + 1 == pxs){
				buff_out[ (j + 1) * bxs * n_size - 1] = '\n';
			}
		}
		for(i = 0; i < (bxs) * (bys) * n_size ; ++i){
			if (buff_out[i] == '\0'){
				buff_out[i] = ' ';
			}
		}
		MPI_File fp;
		MPI_Datatype filetype,datatype;
		MPI_Type_vector(bys, bxs * n_size, bxs * pxs * n_size, MPI_CHAR, &filetype);
		MPI_Type_commit(&filetype);
		MPI_Type_create_hvector(bys, bxs*n_size, bxs*n_size*sizeof(char), MPI_CHAR, &datatype);
		MPI_Type_commit(&datatype);
		MPI_File_delete(out_str.c_str(), MPI_INFO_NULL);
		MPI_File_open(my_comm, out_str.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
		MPI_File_set_view(fp, (y * bys * bxs * pxs + x*bxs)*n_size*sizeof(char), MPI_CHAR, filetype, "native", MPI_INFO_NULL);
		MPI_File_write_all(fp, buff_out, 1, datatype, MPI_STATUS_IGNORE);
		MPI_File_close(&fp);
		
		MPI_Comm_free(&my_comm);
		free(data);
	}
	
	MPI_Finalize();

	return 0;
}