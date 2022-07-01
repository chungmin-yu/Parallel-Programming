__kernel void convolution(__global char * filter, int filterWidth, 
                          int imageHeight, int imageWidth,
                          __global float * inputImage, __global float * outputImage)
{    
    int halffilterSize = filterWidth / 2;
    int ix = get_global_id(0);
    int iy = get_global_id(1);    
    int sum = 0;
    int idxI = 0;
    int idxF = 0;

    int k_start = -halffilterSize + iy >= 0 ? -halffilterSize : 0;
    int k_end = halffilterSize + iy < imageHeight ? halffilterSize : halffilterSize + iy - imageHeight - 1;
    int l_start = -halffilterSize + ix >= 0 ? -halffilterSize : 0;
    int l_end = halffilterSize + ix < imageWidth ? halffilterSize : halffilterSize + ix - imageWidth - 1;

    for (int k = k_start; k <= k_end; k++){
	idxI = (iy + k) * imageWidth + ix;
	idxF = (k + halffilterSize) * filterWidth + halffilterSize;
        for (int l = l_start; l <= l_end; l++){
	    sum += inputImage[idxI + l] * filter[idxF + l];
	}
    }
    outputImage[iy * imageWidth + ix] = sum;
}
