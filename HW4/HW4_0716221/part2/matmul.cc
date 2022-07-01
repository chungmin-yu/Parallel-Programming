# include <stdlib.h>
# include <stdio.h>
# include <mpi.h>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,	int **a_mat_ptr, int **b_mat_ptr){
    int world_size, world_rank;
    int *ptr;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_rank == 0){
		scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
		int asize = (*n_ptr)*(*m_ptr);
		int bsize = (*m_ptr)*(*l_ptr);
		*a_mat_ptr = (int*)calloc(asize, sizeof(int));
		*b_mat_ptr = (int*)calloc(bsize, sizeof(int));

		for (int i = 0; i < *n_ptr; i++){
	    	for (int j = 0; j < *m_ptr; j++){
				ptr = *a_mat_ptr + i * (*m_ptr) + j;
				scanf("%d", ptr);
		    }
		}

		for (int i = 0; i < *m_ptr; i++){
		    for (int j = 0; j < *l_ptr; j++){
				ptr = *b_mat_ptr + i * (*l_ptr) + j;
				scanf("%d", ptr);
	    	}
		}

	}
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    
    int world_size, world_rank;    
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int num_request = world_size - 1;
    int div = n / num_request;
    int mod = n % num_request;  
    int rows, offset, NN, MM, LL;
   
    if (world_rank == 0){
		int *c = (int*)calloc(n*l, sizeof(int));	
		offset = 0;

		// send matrix data  
		for (int destination = 1; destination <= num_request; destination++){
	        MPI_Send(&n, 1, MPI_INT, destination, 1, MPI_COMM_WORLD);
	        MPI_Send(&m, 1, MPI_INT, destination, 1, MPI_COMM_WORLD);
	        MPI_Send(&l, 1, MPI_INT, destination, 1, MPI_COMM_WORLD);
	        rows = (destination <= mod)? div + 1: div;
	        MPI_Send(&offset, 1, MPI_INT, destination, 1, MPI_COMM_WORLD);
	        MPI_Send(&rows, 1, MPI_INT, destination, 1, MPI_COMM_WORLD);	        	        
	        MPI_Send(&a_mat[offset * m], rows * m, MPI_INT, destination, 1, MPI_COMM_WORLD);
	        MPI_Send(&b_mat[0], m * l, MPI_INT, destination, 1, MPI_COMM_WORLD);
	        offset += rows;
	    }

	    // receive results
	    for (int i = 1; i <= num_request; i++){
	    	MPI_Recv(&offset, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
	        MPI_Recv(&rows, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);	        	        
	        MPI_Recv(&c[offset * l], rows * l, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
	    }
	
		// print results	
		for(int i=0;i<n;++i){
	    	for(int j=0;j<l;++j){
	        	printf("%d ", c[i * l + j]);
	    	}
	    	printf("\n");
		}


		free(c);

    }else{
        MPI_Recv(&NN, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&MM, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&LL, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

    	int *a = (int*)calloc(NN*MM, sizeof(int));
    	int *b = (int*)calloc(MM*LL, sizeof(int));
    	int *c = (int*)calloc(NN*LL, sizeof(int));

    	MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);        
        MPI_Recv(&a[0], rows * MM, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], MM * LL, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        // calculate partial multiplication
        for (int i = 0; i < LL; i++){
            for (int j = 0; j < rows; j++){
        		c[j * LL + i] = 0;
        		for (int k = 0; k < MM; k++){
        	    	c[j * LL + i] += a[j * MM + k] * b[k * LL + i];
        		}
            }
        }

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);       
        MPI_Send(&c[0], rows * LL, MPI_INT, 0, 2, MPI_COMM_WORLD);

		free(a);
    	free(b);
		free(c);
    }
}

void destruct_matrices(int *a_mat, int *b_mat){
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    
    if (world_rank == 0){
		free(a_mat);
		free(b_mat);
    }
}
