#include "mpi.h"
#include <stdio.h>
int main(int argc,char *argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	//int x = 4, y =6;
	int x,y;
	printf("enter x");
	scanf("%d",&x);

	printf("enter y");
	scanf("%d",&y);
	
	switch(rank){
	case 0:
		printf("Rank : %d Multiplication answer is -- %d\n",rank,x*y);
		break;
	case 1:
		printf("Rank : %d Addition answer is -- %d\n",rank,x+y);
		break;
	case 2:
		printf("Rank : %d Subtract answer is -- %d\n",rank,x-y);
		break;
	case 3:
		printf("Rank : %d Division answer is -- %d\n",rank,x/y);
		break;
	}
	MPI_Finalize();
	return 0;
}