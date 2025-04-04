#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int x = 5;
	int ans = pow(5,rank);
	printf("for rank %d ans is %d",rank,ans);
	MPI_Finalize();
	return 0;
}