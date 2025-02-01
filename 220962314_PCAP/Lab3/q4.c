#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]){
	int rank, size,N,B[10],c,i;

	char A[100],A1[100];
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	if(rank==0){
		N=size;

		fprintf(stdout,"Enter a string");
		fflush(stdout);
		scanf("%s",A);
		scanf("%s",A1);
	}
	int len=strlen(A);
	char C[100],C1[100];
	MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Scatter(A,len,MPI_CHAR,C,len,MPI_CHAR,0,MPI_COMM_WORLD);
	MPI_Scatter(A1,len,MPI_CHAR,C1,len,MPI_CHAR,0,MPI_COMM_WORLD);
	// fprintf(stdout,"I have received %d in process %d\n",c,rank);
	// fflush(stdout);
	char ans[1000]="";
    for (i = 0; i < len; i++) {
        ans[2*i] = C[i];     
        ans[2*i + 1] = C1[i]; 
    }
    ans[2*len] = '\0'; // Null terminate the string	printf("%s",ans);
	MPI_Gather(ans,len*2,MPI_CHAR,B,len*2,MPI_CHAR,0,MPI_COMM_WORLD);
	if(rank==0){
		fprintf(stdout,"The result gathered in the root\n");
		fflush(stdout);
		// char sum = "";
		printf("ans is %s",B);
		// for(i = 0;i<strlen(B);i++){
		// 	sum+=B[i];
		// }
		// printf("Final total number of vowels are == %d",sum);
	}
	MPI_Finalize();
	return 0;
}