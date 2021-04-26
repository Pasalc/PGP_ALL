#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string>
#include <limits>
#include <unistd.h>
#include <omp.h>
#include <cmath>
#include <chrono>
#define TIMEOUT_TIME 5

double Unext(double* u, int i, int j, double hxs, double hys, int linesize){
	double hyd=1/(hxs*hxs);
	double hxd=1/(hys*hys);
	return ((u[(i+1)*linesize+j]+u[(i-1)*linesize+j])*hxd+(u[i*linesize+j+1]+u[i*linesize+j-1])*hyd) / (2*(hxd+hyd));
}

double Calc_local_eps(double* u,double* uprev, int bxs,int bys){
	int i,j; 
	double maxe=0;
	int linesize=bxs+2;
	#pragma omp parallel for private(j) reduction (max: maxe)
	for(i=1;i<bys+1;++i){
		for(j=1;j<bxs+1;++j){
			maxe = std::max(maxe,std::abs(u[i*linesize+j]-uprev[i*linesize+j]));
		}
	}
	return maxe;
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

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
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
	int n=(bxs+2)*(bys+2);
	double hxs=lx/(bxs*pxs);
	double hys=ly/(bys*pys);
	int linesize=bxs+2;
	
	int color = wrank<pys*pxs ? 0: MPI_UNDEFINED;
	MPI_Comm my_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &my_comm);
	if(color==0){
		char *buf, *buf_p;
		int bsize;
		double* dataswap = (double*)malloc(sizeof(double) * n);
		double* data = (double*)malloc(sizeof(double) * n);
		int rank, nprocs; 
		MPI_Comm_size(my_comm, &nprocs);
		MPI_Comm_rank(my_comm, &rank);
		MPI_Datatype vertvect;
		MPI_Type_vector(bys, 1, bxs+2, MPI_DOUBLE, &vertvect);
		MPI_Type_commit(&vertvect);
	//Init
		int x=rank%pxs,y=rank/pxs;
		int up,down,left,right;
		right = (x+1== pxs)? -1 : y*pxs + (x + 1) % pxs;
		left = (x-1==-1) ? -1 : y*pxs + (x - 1) % pxs;
		up= (y-1==-1) ? -1 :((y-1)*pxs) + x;
		down = (y+1==pys)? -1: ((y+1)*pxs) + x;
		#pragma omp parallel for
		for(i = 0; i < n; i++){
			data[i] = uzero;//can break? cause int init outside?
		}
		//Init buf
		{
			int lrs, uds;
			MPI_Pack_size(bys,MPI_DOUBLE,my_comm,&lrs);
			MPI_Pack_size(bxs,MPI_DOUBLE,my_comm,&uds);
			int bufsize = 4*MPI_BSEND_OVERHEAD + lrs + uds;
			buf = (char *)malloc( bufsize );
			MPI_Buffer_attach(buf, bufsize);
		}
		
		//Init margins
		if(down==-1){
			for(i = (bys+1)*(bxs+2); i < (bys+2)*(bxs+2); ++i){
				data[i] = ud; 
				dataswap[i]=ud;
			}
		}
		if(up==-1){
			for(i = 0; i < bxs+2; ++i){
				data[i] = uu; 
				dataswap[i]=uu;
			}
		}
		if(left==-1){
			for(i = 0; i < bys+2; ++i){
				data[i*(bxs+2)] = ul;
				dataswap[i*(bxs+2)] = ul;
			}
		}
		if(right==-1){
			for(i = 0; i < bys+2; ++i){
				data[(bxs+1)+i*(bxs+2)] = ur;
				dataswap[(bxs+1)+i*(bxs+2)] = ur;
			}
		}
		//fprintf(stderr, "%d: %d %d %d %d \n", rank, left , right, up , down);
	//main
		if(rank==0)
			fprintf(stderr, "pxs %d,pys %d,bxs %d,bys %d,mineps %lf,lx %lf,ly %lf,ul %lf,ur %lf, uu %lf , ud %lf , uzero %lf, out_str: %s \n", pxs,pys,bxs,bys,mineps,lx,ly,ul,ur,uu,ud,uzero,out_str.c_str());
		int count=0;
		fprintf(stderr, "%d %d %d %d\n",left, right, down, up);
		while(count<100000){
			MPI_Request req[4];
			MPI_Status statuses[4];
			//main
			#pragma omp parallel for private(j)
			for(i=1;i<bys+1;++i){
				for(j=1;j<bxs+1;++j){
					dataswap[i*linesize+j]=Unext(data,i,j,hxs,hys,linesize);
				}
			}
			std::swap(data,dataswap);
			count++;
			//calc eps
			leps=Calc_local_eps(data,dataswap,bxs,bys);
			//fprintf(stderr, "%d,%d: after leps\n", rank,count);
			//reduce eps
			MPI_Allreduce(MPI_IN_PLACE,&leps,1,MPI_DOUBLE,MPI_MAX,my_comm);
			if(rank==0){
				//fprintf(stderr, "leps: %lf\n",leps);
			}
			if(leps<mineps){
				break;
			}
			//fprintf(stderr, "%d,%d: after allred\n", rank,count);
			//!done
			//recv/send from up to down
			int req_count=0;
			if(left!=-1){
				MPI_Bsend((void*)(data+(bxs+2)+1), 1, vertvect, left, 0, my_comm);
				MPI_Irecv((void*)(data+(bxs+2)), 1, vertvect,left,0,my_comm,&req[req_count]);
				++req_count;
			}
			if(right!=-1){
				MPI_Bsend(data+(bxs+2)*2-2, 1, vertvect, right, 0, my_comm);
				MPI_Irecv(data+(bxs+2)*2-1, 1, vertvect, right, 0, my_comm,&req[req_count]);
				++req_count;
			}
			if(up!=-1){
			
				MPI_Bsend(data+(bxs+2)+1, bxs, MPI_DOUBLE, up, 0, my_comm);
				MPI_Irecv(data+1, bxs, MPI_DOUBLE, up, 0, my_comm,&req[req_count]);
				++req_count;
				}
			if(down!=-1){
				MPI_Bsend(data+(bys)*(bxs+2)+1, bxs, MPI_DOUBLE, down, 0, my_comm);
				MPI_Irecv(data+(bys+1)*(bxs+2)+1, bxs, MPI_DOUBLE, down, 0, my_comm,&req[req_count]);
				++req_count;
			}
			//fprintf(stderr, "%d,%d: wait\n", rank,count);
			MPI_Waitall(req_count, req,statuses);
			//fprintf(stderr, "%d,%d: barrier\n", rank,count);
			//MPI_Barrier(my_comm);//not needed
			//fprintf(stderr, "%d,%d: done\n", rank,count);
		}
		MPI_Buffer_detach( &buf_p, &bsize );
	//Write
		int n_size = 15;
		char * buff_out = (char *) malloc(sizeof(char) * (bxs) * (bys)*n_size);
		memset(buff_out, ' ', (bxs) * (bys) * n_size * sizeof(char));
		fprintf(stderr, "count : %d\n",count);
		
		#pragma omp parallel for private(i)
		for(j = 0; j < bys; ++j) {
			for(i = 0; i < bxs; ++i){
				sprintf(buff_out + (j * bxs + i)*n_size, " %.6e ", data[(j+1) * (bxs+2) + (i+1)]);
			}
			if (x + 1 == pxs){
				buff_out[ (j + 1) * bxs * n_size - 1] = '\n';
			}
		}
		#pragma omp parallel for
		for(i = 0; i < (bxs) * (bys) * n_size ; ++i){
			if (buff_out[i] == '\0'){
				buff_out[i] = ' ';
			}
		}
		MPI_Barrier(my_comm);
		std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[ms](" <<rank<<")"<< std::endl;
		MPI_File fp;
		MPI_Datatype filetype,datatype;
		MPI_Type_vector(bys, bxs * n_size, bxs * pxs * n_size, MPI_CHAR, &filetype);
		MPI_Type_commit(&filetype);
		MPI_Type_vector(bys, bxs*n_size, bxs*n_size, MPI_CHAR, &datatype);
		MPI_Type_commit(&datatype);
		MPI_File_delete(out_str.c_str(), MPI_INFO_NULL);
		MPI_File_open(my_comm, out_str.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
		MPI_File_set_view(fp, (y * bys * bxs * pxs + x*bxs)*n_size*sizeof(char), MPI_CHAR, filetype, "native", MPI_INFO_NULL);
		MPI_File_write_all(fp, buff_out, 1, datatype, MPI_STATUS_IGNORE);
		MPI_File_close(&fp);
		
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		if(rank==0){
			std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[ms]" << std::endl;
		}
		MPI_Comm_free(&my_comm);
		free(data);
	}
	if(wrank==0){
		fprintf(stderr, "Program end\n" );
	}
	MPI_Finalize();
	return 0;
}