#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
    int rank;
    char word[100];  
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status stat;

    if(rank == 0) {
        printf("\nEnter the word: ");
        scanf("%s", word);

        int len = strlen(word);
        MPI_Ssend(&len, 1, MPI_INT, 1, 10, MPI_COMM_WORLD);  
        MPI_Ssend(word, len, MPI_CHAR, 1, 10, MPI_COMM_WORLD);

        printf("\nRank 0: Sent word, now awaiting toggled word...\n");
        MPI_Recv(word, len, MPI_CHAR, 1, 10, MPI_COMM_WORLD, &stat);
        printf("\nRank 0: Toggled word: %s\n", word);
    }
    else {
        int len = 0;
        MPI_Recv(&len, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &stat);
        // char* word = (char*) malloc(len * sizeof(char));
        printf("len is %d \n",len); 
        char word[100];
        MPI_Recv(word, len, MPI_CHAR, 0, 10, MPI_COMM_WORLD, &stat);
        // printf("%s\n",word);
        for(int i = 0; i < len; i++) {
        	// printf("Toggling - %c at %d\n",word[i],i);
            if(word[i] >= 'A' && word[i] <= 'Z') {
                word[i] += 32; 
            }
            else {
                word[i] -= 32;
            }
        }

        printf("Rank 1: Toggled word to be sent: %s\n", word);
        MPI_Ssend(word, len, MPI_CHAR, 0, 10, MPI_COMM_WORLD); 

    }

    MPI_Finalize();
    return 0;
}
