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
		int n;
		fprintf(stdout,"Enter n");
		scanf("%d",&n);
		int* array = (int*)calloc(n, sizeof(int));
		for(int i = 0;i<n;i++){

			scanf("%d",&array[i]);
		}
		fprintf(stdout, "\nRank 0 sending...\n");
		for(int i = 1; i < n; i++) {
			MPI_Send(array+i, 1, MPI_INT, i, 10, MPI_COMM_WORLD);
		}
		fflush(stdout);
	}
	else {
		int x;
		MPI_Recv(&x, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &stat);
		if(x%2==0){
			printf("Rank %d value is %d\n",rank,x*x);
		}else{
			printf("Rank %d value is %d\n",rank,x*x*x);
		}
		// fprintf(stdout, "Rank %d:\tReceived %d\n", rank, x);
	}
	MPI_Finalize();
	exit(0);
}