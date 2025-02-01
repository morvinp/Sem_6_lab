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
		fprintf(stdout, "\nRank 0 sending...\n");
		for(int i = 1; i < nop; i++) {
			int x = i*10;
			MPI_Send(&x, 1, MPI_INT, i, 10, MPI_COMM_WORLD);
		}
	}
	else {
		int x;
		MPI_Recv(&x, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &stat);
		fprintf(stdout, "Rank %d:\tReceived %d\n", rank, x);
	}
	MPI_Finalize();
	exit(0);
}