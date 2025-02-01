#include <stdio.h>
#include "mpi.h"
int main(int argc,char* argv[]){
	int rank, size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	int ans[4][4];
	int factans[4];
	int result[4],r[4][4];
	if(rank==0){
		fprintf(stdout,"Enter the matrix values");
		fflush(stdout);
		for(int i =0;i<4;i++){
			for(int j = 0;j<4;j++){
				scanf("%d",&ans[i][j]);
			}
		}
		// ans = {{1,2,3,4},{1,2,3,1},{1,1,1,1},{2,1,2,1}};
	}
	MPI_Scatter(ans,4,MPI_INT,factans,4,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Scan(factans,result,4,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Gather(result,4,MPI_INT,r,4,MPI_INT,0,MPI_COMM_WORLD);
	// MPI_Reduce(&factans,&factsum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

	if(rank ==0){
		for(int i =0;i<4;i++){
			for(int j = 0;j<4;j++){
				printf("%d\t",r[i][j]);
			}
			printf("\n");
		}
	}
	MPI_Finalize();
	return 0;
}
