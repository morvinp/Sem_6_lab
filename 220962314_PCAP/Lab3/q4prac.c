#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char s1[100] = {0}, s2[100] = {0};
    char one, two;
    char concated[3] = {0};  // Initialize to zeros directly
    char *ans = NULL;
    
    if (rank == 0) {
        printf("Enter first string: ");
        scanf("%s", s1);
        printf("Enter second string: ");
        scanf("%s", s2);
        
        // Allocate space and initialize to zero using array initialization
        ans = (char *)calloc(2 * strlen(s1) + 1, sizeof(char));
    }
    
    // Broadcast the string lengths to all processes
    int len;
    if (rank == 0) {
        len = strlen(s1);
    }
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Scatter characters from both strings
    MPI_Scatter(s1, 1, MPI_CHAR, &one, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(s2, 1, MPI_CHAR, &two, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // Combine characters
    concated[0] = one;
    concated[1] = two;
    
    // Gather the results
    MPI_Gather(concated, 2, MPI_CHAR, ans, 2, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Interleaved result: %s\n", ans);
        free(ans);
    }
    
    MPI_Finalize();
    return 0;
}