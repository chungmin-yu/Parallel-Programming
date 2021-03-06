#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    //cl_int status;
    char *charFilter = (char *)malloc(filterWidth * filterWidth * sizeof(char));
    int new_filterWidth = filterWidth;
    int checkStart = 0;
    int checkEnd = filterWidth - 1;
    int check = 1;
    while(check == 1 && checkStart < checkEnd) {
        for (int i = 0; i < filterWidth && check == 1; i++) 
		if(filter[checkStart * filterWidth + i] != 0) check = 0;  // upper
        for (int i = 0; i < filterWidth && check == 1; i++) 
		if(filter[checkEnd * filterWidth + i] != 0) check = 0;  // lower
        for (int i = 0; i < filterWidth && check == 1; i++) 
		if(filter[i * filterWidth + checkStart] != 0) check = 0;  // left
        for (int i = 0; i < filterWidth && check == 1; i++) 
		if(filter[i * filterWidth + checkEnd] != 0) check = 0;  // right
        if (check == 1) new_filterWidth -= 2;
        checkStart++;
        checkEnd--;
    }
    int charFilter_start = (filterWidth - new_filterWidth) % 2 == 0 ? (filterWidth - new_filterWidth) / 2 : 0;
    for (register int i = 0; i < new_filterWidth; i++)
        for (register int j = 0; j < new_filterWidth; j++)
            charFilter[i * new_filterWidth + j] = filter[((charFilter_start + i) * filterWidth) + charFilter_start + j];

    // declare the size
    int filterSize = new_filterWidth * new_filterWidth * sizeof(char);
    int inputSize = imageHeight * imageWidth * sizeof(float);

    // create kernel function
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    // create command queue
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, NULL);

    // allocate device memory    
    cl_mem bufInputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY, inputSize, NULL, NULL);
    cl_mem bufOutputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, inputSize, NULL, NULL);
    cl_mem bufFilter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, NULL);

    // copy data from host memory to device memory
    clEnqueueWriteBuffer(queue, bufFilter, 0, 0, filterSize, (void *)charFilter, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufInputImage, 0, 0, inputSize, (void *)inputImage, 0, NULL, NULL);

    // set kernel function args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufFilter);
    clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&new_filterWidth);    
    clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bufInputImage);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&bufOutputImage);

    // workgroups parameter
    size_t localws[2] = {8, 8};
    size_t globalws[2] = {imageWidth, imageHeight};

    // execute kernel function
    clEnqueueNDRangeKernel(queue, kernel, 2, 0, globalws, localws, 0, NULL, NULL);

    // copy data from device memory to host memory
    clEnqueueReadBuffer(queue, bufOutputImage, 0, 0, inputSize, (void *)outputImage, NULL, NULL, NULL);

    // release opencl object
    //clReleaseKernel(kernel);
    //clReleaseCommandQueue(queue);    
    //clReleaseMemObject(bufInputImage);
    //clReleaseMemObject(bufOutputImage);
    //clReleaseMemObject(bufFilter);

    free(charFilter);
    
}
