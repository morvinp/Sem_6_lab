#include <stdio.h>
#include "mpi.h"

int main(int argc,char* argv[]){
	int rank, size, fact = 1,factsum, i;
	MPI_Init(&argc,&argv);
	int ans[3][3];
	int ele;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if(rank==0){
		fprintf(stdout,"Enter the matrix values");
		fflush(stdout);
		for(int i = 0;i<3;i++){
			for(int j = 0;j<3;j++){
				int val;
				scanf("%d", &val);
				ans[i][j]=val;
			}
		}
		fprintf(stdout,"Enter an element");
		fflush(stdout);
		scanf("%d",&ele);
	}
	MPI_Bcast(&ele,1,MPI_INT,0,MPI_COMM_WORLD);

	int onerow[3];
	int count =0;
	int result;
	MPI_Scatter(ans,3,MPI_INT,onerow,3,MPI_INT,0,MPI_COMM_WORLD);
	for(int i = 0;i<3;i++){
		// printf("value recieved is %d",onerow[i]);
		if(onerow[i]==ele){
			count++;
		}
	}
	MPI_Reduce(&count,&result,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	// int factans;
	// MPI_Scan(&ans,&factans,1,MPI_INT,MPI_PROD,MPI_COMM_WORLD);
	// MPI_Reduce(&factans,&factsum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

	if(rank ==0){
		printf("Sum is %d\n",result);

	}
	MPI_Finalize();
	return 0;
}