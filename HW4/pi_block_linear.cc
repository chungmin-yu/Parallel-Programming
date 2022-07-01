#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;
    unsigned int seed = time(NULL)*world_rank;
    long long int total = 0;
    long long int local_count = 0;
    long long int num_task = tosses/world_size;
    double x, y, z;

    MPI_Barrier(MPI_COMM_WORLD);
    srand(seed);
    while(num_task--){
	x = (double)rand_r(&seed)/RAND_MAX;
	y = (double)rand_r(&seed)/RAND_MAX;
	z = x*x+y*y;
	if(z <= 1.0){
		local_count++;
	}
	
    }

    if (world_rank > 0)
    {
        // TODO: handle workers
	MPI_Send(&local_count, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
	total = local_count;
	for(int source = 1 ; source < world_size ; source++){
		MPI_Recv(&local_count, 1, MPI_LONG_LONG_INT, source, 0, MPI_COMM_WORLD, &status);
		total += local_count;
	}
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
	double pi_result = 4.0 * total/(double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
