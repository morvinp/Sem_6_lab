#include <mpi.h>
#include <stdio.h>

int main(int argc, char * argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	int c,A[100];
	if(rank ==0){
		int n;
		printf("Enter %d values", size);
		for(int i = 0;i<size;i++){
			scanf("%d", &A[i]);
		}
	}
	MPI_Scatter(A,1,MPI_INT,&c,1,MPI_INT,0,MPI_COMM_WORLD);
	int ans = 1;
	for(int i = 2;i<=c;i++){
		ans*=i;
	}
	int B[100];
	MPI_Gather(&ans,1,MPI_INT,B,1,MPI_INT,0,MPI_COMM_WORLD);
	if(rank==0){
		printf("\nthis is what was gathered --\n");
		for(int i = 0;i<size;i++){
			printf("%d \t",B[i]);
		}
	}
	MPI_Finalize();
	return 0;
}