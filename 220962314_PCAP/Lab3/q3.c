#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]){
	int rank, size,N,B[10],c,i;
	char A[100];
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	if(rank==0){
		N=size;
		// fprintf(stdout,"Enter %d values:\n",N);
		// fflush(stdout);
		// for(i =0;i<N;i++){
		// 	scanf("%d",&A[i]);
		// }
		fprintf(stdout,"Enter a string");
		fflush(stdout);
		scanf("%s",A);
	}
	int len=strlen(A);
	char C[100];
	MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Scatter(A,len,MPI_CHAR,C,len,MPI_CHAR,0,MPI_COMM_WORLD);
	// fprintf(stdout,"I have received %d in process %d\n",c,rank);
	// fflush(stdout);
	int ans = 0;
	for(int i = 0;C[i]!='\0';i++){
		if(C[i]=='a' || C[i]=='e' || C[i]=='i' || C[i]=='o' || C[i]=='u'){
			ans+=0;
		}else{
			ans+=1;
		}
	}

	MPI_Gather(&ans,1,MPI_INT,B,1,MPI_INT,0,MPI_COMM_WORLD);
	if(rank==0){
		fprintf(stdout,"The result gathered in the root\n");
		fflush(stdout);
		int sum = 0;
		for(i = 0;i<N;i++){
			sum+=B[i];			
		}
		printf("Final total number of vowels are == %d",sum);
	}
	MPI_Finalize();
	return 0;
}