#include "mpi.h"
#include <stdio.h>
int main(int argc,char *argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	// printf("My rank is %d in total %d processes",rank,size);
	char word[] = "HELLO";
	char before = word[rank];
	char after = word[rank]+32;
	word[rank]+=32;
	printf("Toggled position %d for rank %d before : %c after : %c word is now %s\n",rank,rank,before,after,word);
	MPI_Finalize();
	return 0;
}