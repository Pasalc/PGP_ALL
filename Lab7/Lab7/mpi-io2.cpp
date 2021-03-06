#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
	int i, j, rank, nprocs, nx = 4, ny = 5;
	int *data = (int *)malloc(sizeof(int) * nx * ny);
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	fprintf(stderr, "%d(%d)\n", rank, nprocs);

/*	for(j = 0; j < ny; j++)
		for(i = 0; i < nx; i++)
			data[j * nx + i] = rank;//j * (nx * nprocs) + rank * nx + i; 
*/
	int bxs=3,bys=3;
	MPI_File fp;
	MPI_Datatype filetype,datatype;
	MPI_Type_vector(bys, bxs, bxs * nprocs, MPI_DOUBLE, &filetype);
	MPI_Type_commit(&filetype);
	MPI_Type_vector(bys, bxs, 2, MPI_DOUBLE, &datatype);
	MPI_Type_commit(&datatype);
	MPI_Datatype filetype;

//	MPI_File_delete("data", MPI_INFO_NULL);
//	MPI_File_open(MPI_COMM_WORLD, "data", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_File_open(MPI_COMM_WORLD, "out.txt", MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);

	MPI_File_set_view(fp, bxs * rank * sizeof(double), MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

//	MPI_File_write_all(fp, data, nx * ny, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_read_all(fp, data, bxs * bys, MPI_DOUBLE, MPI_STATUS_IGNORE);

	printf("%d: ", rank);
	for(i = 0; i < nx * ny; i++)
		printf("%d ", data[i]);
	printf("\n");
	
	MPI_File_close(&fp);

	MPI_Finalize();
	free(data);

	return 0;
}
