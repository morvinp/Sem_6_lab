#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
	int rank, nop;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nop);
	MPI_Status stat;
	if(rank == 0) {
		fprintf(stdout, "\nEnter %d integers:\n", nop-1);
		int* arr = (int*) calloc(nop-1, sizeof(int));
		for(int i = 0; i < nop-1; i++) {
			scanf("%d", &arr[i]);
		}
		int bsize = sizeof(int) + MPI_BSEND_OVERHEAD;
		int* buf = (int*) malloc(bsize*sizeof(int));
		printf("Attaching Buffer\n");
		MPI_Buffer_attach(buf, bsize);
		for(int i = 1; i < nop; i++) {
			MPI_Bsend(&arr[i-1], 1, MPI_INT, i, 10, MPI_COMM_WORLD);
		}
		printf("Detatching Buffer\n");
		MPI_Buffer_detach(&buf, &bsize);
	}
	else {
		int x;
		MPI_Recv(&x, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &stat);
		if(rank%2 == 0) {
			x *= x;
		}
		else {
			x *= x*x;
		}
		fprintf(stdout,"Rank %d:\t %d\n", rank, x);
	}
	MPI_Finalize();
	exit(0);
}