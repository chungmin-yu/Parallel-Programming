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

    MPI_Win win;

    // TODO: MPI init
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

    if (world_rank == 0)
    {
        // Master
	MPI_Alloc_mem(sizeof(long long int), MPI_INFO_NULL, &total);
	MPI_Win_create(&total, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
	total = local_count;
    }
    else
    {
        // Workers
	MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
	MPI_Accumulate(&local_count, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);
       	MPI_Win_unlock(0, win);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
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
