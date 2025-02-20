#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char * argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	int ans = rank+1;
	int factans,factsum;
	MPI_Scan(&ans, &factans,1,MPI_INT,MPI_PROD,MPI_COMM_WORLD);
	MPI_Reduce(&factans,&factsum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

	if(rank==0)
	printf("The answer is %d", factsum);

	MPI_Finalize();
	return 0;
}