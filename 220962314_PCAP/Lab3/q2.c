#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]){
	int rank, size,N,A[1000],c,i;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	int M;
	float B[1000];
	if(rank==0){
		N=size;
		fprintf(stdout,"Enter value of M\n");
		scanf("%d",&M);
		fprintf(stdout,"Enter %d values:\n",N*M);
		fflush(stdout);
		for(i =0;i<N*M;i++){
			scanf("%d",&A[i]);
		}
	}
	int C[1000];
	MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Scatter(A,M,MPI_INT,C,M,MPI_INT,0,MPI_COMM_WORLD);
	// fprintf(stdout,"I have received %d in process %d\n",C[1],rank);
	fflush(stdout);

	float ans = 0;
	for(i = 0;i<M;i++){
		ans+=C[i];
		// printf("Value of c is === %d and ans is %f",C[i],ans);
	}
	// printf("%d and %d",ans,M);
	ans = ans/M;
	// printf("\n this is the valuye of ans %f\n",ans);
	MPI_Gather(&ans,1,MPI_FLOAT,B,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	if(rank==0){
		fprintf(stdout,"The result gathered in the root\n");
		fflush(stdout);
		float f = 0;
		for(i = 0;i<N;i++){
			f+=B[i];
		}
		fprintf(stdout,"Final Answer is = %f",f/N);
		fflush(stdout);
	}
	MPI_Finalize();
	return 0;
}