#include <stdio.h>
#include "mpi.h"
#include <string.h>
int main(int argc,char* argv[]){
	int rank, size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Status stat;
	char str[1000];
	if(rank==0){
		fprintf(stdout,"Enter the string");
		fflush(stdout);
		scanf("%s",str);
	}
	char val[1];
	MPI_Scatter(str,1,MPI_CHAR,val,1,MPI_CHAR,0,MPI_COMM_WORLD);
	// MPI_Scan(factans,result,4,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	// MPI_Gather(result,4,MPI_INT,r,4,MPI_INT,0,MPI_COMM_WORLD);
	// MPI_Reduce(&factans,&factsum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	int len = rank+1;
	// printf("%s\t at rank %d",val,rank);
	char ans[rank+2];
	for(int i = 0;i<rank+1;i++){
		// printf("%s\n",ans);
		ans[i]=val[0];
	
	}
	// printf("%s is the ans", ans);
	char result[100];
	for(int i=0;i<size;i++){
		// MPI_Send(&len,1,MPI_INT,i,i,MPI_COMM_WORLD);
		// printf("I am sending %s at %d\n",ans,rank);
		MPI_Send(&len,1,MPI_INT,0,i,MPI_COMM_WORLD);
		MPI_Send(ans,rank+1,MPI_CHAR,0,i,MPI_COMM_WORLD);

	}
	char res[100];
	int lens;
	if(rank ==0){
		for(int i = 0;i<size;i++){
			MPI_Recv(&lens,1,MPI_INT,i,i,MPI_COMM_WORLD,&stat);
			MPI_Recv(res,len,MPI_CHAR,i,i,MPI_COMM_WORLD,&stat);
			// printf("i am getting %s at rank %d",res,rank);
			// strcat(result,res);
			printf("Answer is %s\n",res);
		}
		printf("%s is the answer hopefully", result);
	}
	MPI_Finalize();
	return 0;
}
