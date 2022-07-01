#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 25

__global__ void mandelKernel(int* data, float lowerX, float lowerY, float stepX, float stepY, int maxIteration, int width) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
   
    
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;
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

    int idx = thisX + thisY * width;
    data[idx] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int size = resX * resY * sizeof(int);

    int *d = (int*) malloc(size);
    int *data;
    cudaMalloc(&data, size);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(resX/threadsPerBlock.x, resY/threadsPerBlock.y);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(data, lowerX, lowerY, stepX, stepY, maxIterations, resX);

    cudaMemcpy(d, data, size, cudaMemcpyDeviceToHost);
    memcpy(img, d, size);
    cudaFree(data);
    free(d);

}
