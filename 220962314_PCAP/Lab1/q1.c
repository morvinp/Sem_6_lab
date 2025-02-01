/*student@dbl-08:~/Documents/220962314/Lab1$ mpicc q1.c -o o.out -lm
student@dbl-08:~/Documents/220962314/Lab1$ mpirun -np 4 ./o.out
For rank : 0 ans is 1
For rank : 1 ans is 5
For rank : 2 ans is 25
For rank : 3 ans is 125
*/


#include "mpi.h"
#include <stdio.h>
#include <math.h>
int main(int argc,char *argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	int x = 5;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	// printf("My rank is %d in total %d processes",rank,size);
	int ans = pow(x,rank);
	printf("For rank : %d ans is %d\n",rank,ans);
	MPI_Finalize();
	return 0;
}