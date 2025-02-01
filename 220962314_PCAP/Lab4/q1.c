#include <stdio.h>
#include "mpi.h"
void ErrorHandler(int error);
int main(int argc,char* argv[]){
	int rank, size, fact = 1,factsum, i;
	MPI_Init(&argc,&argv);
	int err = 33333;
	MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	int error=MPI_Comm_size(MPI_COMM_WORLD,&err);
	ErrorHandler(error);
	// for(i = 1;i<=rank;i++){
	// 	fact = fact*i;
	// }
	int ans = rank+1;
	int factans;
	MPI_Scan(&ans,&factans,1,MPI_INT,MPI_PROD,MPI_COMM_WORLD);
	MPI_Reduce(&factans,&factsum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

	if(rank ==0){
		printf("Sum of all the factorial=%d",factsum);
	}
	MPI_Finalize();
	return 0;
}

void ErrorHandler(int error){
	if(error!=MPI_SUCCESS){
		char error_string[BUFSIZ];
		int length_of_error_string,error_class;
		MPI_Error_class(error,&error_class);
		MPI_Error_string(error,error_string,&length_of_error_string);
		printf("%d %s\n",error,error_string);
		MPI_Error_string(error_class,error_string,&length_of_error_string);
		printf("%d %s\n",error_class,error_string);
	}
}