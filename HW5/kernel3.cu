#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 25
#define GROUP_SIZE 4

__global__ void mandelKernel(int* data, float lowerX, float lowerY, float stepX, float stepY, int maxIteration, int pitch, int scale) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
   
    
    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * scale;
    int thisY = (blockIdx.y * blockDim.y + threadIdx.y) * scale;

    for (int a = 0; a < scale; a++){
	for (int b = 0; b < scale; b++){
    	    float c_re = lowerX + (thisX + b) * stepX;
    	    float c_im = lowerY + (thisY + a) * stepY;
    	    float z_re = c_re, z_im = c_im;
     
	    int i;
    	    for (i = 0; i < maxIteration; ++i){
	        if (z_re * z_re + z_im * z_im > 4.f) 
	    	    break;

	        float new_re = z_re * z_re - z_im * z_im;
	 	float new_im = 2.f * z_re * z_im;
		z_re = c_re + new_re;
		z_im = c_im + new_im;
    	    }

	        int* row = (int*)((char*)data + (thisY + a) * pitch);
                row[thisX + b] = i;
	}
    }	
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int size = resX * resY * sizeof(int);
    size_t pitch = 0; 
    int scale = GROUP_SIZE;

    int *d;
    cudaHostAlloc(&d, size, cudaHostAllocMapped);
    int *data;
    cudaMallocPitch(&data, &pitch, resX * sizeof(int), resY);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(resX/threadsPerBlock.x/scale, resY/threadsPerBlock.y/scale);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(data, lowerX, lowerY, stepX, stepY, maxIterations, pitch, scale);

    cudaMemcpy2D(d, resX * sizeof(int), data, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, d, size);
    cudaFree(data);
    cudaFreeHost(d);

}
