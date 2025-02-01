#include "mpi.h"
#include <stdio.h>

int factorial(int x){
	int ans = 1;
	while(x!=0){
		ans*=x;
		x--;
	}
}

int fibonacci(int x){
	if(x==0){
		return 0;
	}else if (x==1){
		return 1;
	}else{
		return fibonacci(x-1)+fibonacci(x-2);
	}
}
int main(int argc,char *argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	int x = 5;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	// printf("My rank is %d in total %d processes",rank,size);
	if(rank%2==0){
		printf("Factorial of rank %d is %d\n",rank,factorial(rank));
	}else{
		printf("Fibonacci of rank %d is %d\n",rank,fibonacci(rank));
	}
	MPI_Finalize();
	return 0;
}