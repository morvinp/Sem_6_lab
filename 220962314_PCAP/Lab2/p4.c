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
		printf("\nEnter the integer: \n");
		int x; scanf("%d", &x);
		MPI_Send(&x, 1, MPI_INT, rank+1, 10, MPI_COMM_WORLD);
		MPI_Recv(&x, 1, MPI_INT, nop-1, 10, MPI_COMM_WORLD, &stat);
		printf("Rank %d: Received %d. Round Complete!\n", rank, x);
	}
	else {
		int x;
		MPI_Recv(&x, 1, MPI_INT, rank-1, 10, MPI_COMM_WORLD, &stat);
		printf("Rank %d: Received %d\n", rank, x);	x++;
		MPI_Send(&x, 1, MPI_INT, (rank+1) % nop, 10, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	exit(0);
}