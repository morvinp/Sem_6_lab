#include <mpi.h>
#include <stdio.h>
#include <string.h>


int main(int argc, char* argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status stat;
	if(rank==0){
		char s[100];
		scanf("%s", s);
		int len = strlen(s);
		MPI_Ssend(&len, 1, MPI_INT, 1, 10, MPI_COMM_WORLD);
		MPI_Ssend(&s, len, MPI_CHAR, 1,10,MPI_COMM_WORLD);
		char toggled[100];
		MPI_Recv(toggled, len, MPI_CHAR, 1, 10, MPI_COMM_WORLD, &stat);
		printf("\nToggled string is %s\n", toggled);
	}
	else{
		char s[100];
		int len;
		MPI_Recv(&len,1,MPI_INT,0,10,MPI_COMM_WORLD, &stat);
		MPI_Recv(s,len,MPI_CHAR,0,10,MPI_COMM_WORLD,&stat);

		for(int i =0;i<len;i++){
			s[i]-=32;
		}
		MPI_Ssend(s,len,MPI_CHAR,0,10,MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}