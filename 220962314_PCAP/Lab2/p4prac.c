#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Status stat;
	if(rank==0){
		int n;
		scanf("%d", &n);
		printf("\nRank %d recieved %d\n", rank, n);
		n++;
		MPI_Send(&n,1,MPI_INT, 1, 10, MPI_COMM_WORLD);
		int ans;
		MPI_Recv(&ans,1,MPI_INT,rank-1,10,MPI_COMM_WORLD,&stat);
		printf("Finally recieved %d", ans);
	}else{
		int n;
		MPI_Recv(&n,1,MPI_INT,rank-1,10,MPI_COMM_WORLD,&stat);
		printf("\n Rank %d recieved %d\n",rank,n);
		n++;
		MPI_Send(&n,1,MPI_INT,(rank+1)%size,10,MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}