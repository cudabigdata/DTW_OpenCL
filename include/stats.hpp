#ifndef STATS_CUH
#define STATS_CUH

#include <stdio.h>
#include <math.h>
#include "SystemCL.hpp"
#include "parallel_util.hpp"
void znormalize(float * const series, const int L) {
    
    float avg = 0, std=0;
    
    // calculate average
    for (int i = 0; i < L; ++i)
        avg += *(series+i);
    avg /= L;
    
    // calculate standard deviation
    for (int i = 0; i < L; ++i)
        std += (*(series+i))*(*(series+i));
    std = sqrt(std/L - avg*avg);
    
    // z-normalize the input
    for (int i = 0; i < L; ++i) {
        *(series+i) -= avg;
        *(series+i) /= std;
    }
}


void avg_std(cl_mem Series, cl_mem AvgS, cl_mem StdS, int M, int N) {
             
    // create temporary storage for prefix sums
    cl_mem Aprefix;
    cl_mem Sprefix;


    cl_context context = SystemCL::Inst().Context();
    cl_int ret;
    Aprefix = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*(N+1), NULL, &ret);  CLERR(ret);
    Sprefix = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*(N+1), NULL, &ret);  CLERR(ret);

    // convert data type with element-wise copy
    cl_kernel plainCpy = SystemCL::Inst().K_plainCpy;
    ret = clSetKernelArg(plainCpy, 0, sizeof(cl_mem), (void *)&Series);  CLERR(ret)
    ret = clSetKernelArg(plainCpy, 1, sizeof(cl_mem), (void *)&Aprefix); CLERR(ret)
    ret = clSetKernelArg(plainCpy, 2, sizeof(int), (void *)&N);          CLERR(ret)

    size_t global_item_size = GRIDDIM * BLOCKDIM;
    ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(), plainCpy, 1, NULL,
              &global_item_size, NULL, 0, NULL, NULL);  CLERR(ret)

    cl_kernel squareCpy = SystemCL::Inst().K_squareCpy;
    ret = clSetKernelArg(squareCpy, 0, sizeof(cl_mem), (void *)&Series);  CLERR(ret)
    ret = clSetKernelArg(squareCpy, 1, sizeof(cl_mem), (void *)&Sprefix); CLERR(ret)
    ret = clSetKernelArg(squareCpy, 2, sizeof(int), (void *)&N);          CLERR(ret)

    ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(), squareCpy, 1, NULL,
              &global_item_size, NULL, 0, NULL, NULL);

//    // calculate prefix sums
    prefix_sum(Aprefix, N);

    prefix_sum(Sprefix, N);

//    // calculate windowed difference (average)
    cl_kernel window = SystemCL::Inst().K_window;
    ret = clSetKernelArg(window, 0, sizeof(cl_mem), (void *)&Aprefix);  CLERR(ret)
    ret = clSetKernelArg(window, 1, sizeof(cl_mem), (void *)&AvgS);  CLERR(ret)
    ret = clSetKernelArg(window, 2, sizeof(int), (void *)&N);  CLERR(ret)
    ret = clSetKernelArg(window, 3, sizeof(int), (void *)&M);  CLERR(ret)

    global_item_size = GRIDDIM * BLOCKDIM;

    ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(), window, 1, NULL,
              &global_item_size, NULL, 0,NULL, NULL);  CLERR(ret)

//    // calculate windowed difference (standard deviation)
    ret = clSetKernelArg(window, 0, sizeof(cl_mem), (void *)&Sprefix);  CLERR(ret)
    ret = clSetKernelArg(window, 1, sizeof(cl_mem), (void *)&StdS);  CLERR(ret)
    ret = clSetKernelArg(window, 2, sizeof(int), (void *)&N);  CLERR(ret)
    ret = clSetKernelArg(window, 3, sizeof(int), (void *)&M);  CLERR(ret)

    global_item_size = GRIDDIM * BLOCKDIM;
    ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(), window, 1, NULL,
              &global_item_size, NULL, 0, NULL, NULL);  CLERR(ret)


    cl_kernel stddev = SystemCL::Inst().K_stddev;
    ret = clSetKernelArg(stddev, 0, sizeof(cl_mem), (void *)&AvgS);  CLERR(ret)
    ret = clSetKernelArg(stddev, 1, sizeof(cl_mem), (void *)&StdS);  CLERR(ret)
    ret = clSetKernelArg(stddev, 2, sizeof(cl_mem), (void *)&StdS);  CLERR(ret)
    ret = clSetKernelArg(stddev, 3, sizeof(int), (void *)&N);  CLERR(ret)
    ret = clSetKernelArg(stddev, 4, sizeof(int), (void *)&M);  CLERR(ret)
    global_item_size = GRIDDIM * BLOCKDIM;
    ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(), stddev, 1, NULL,
              &global_item_size, NULL, 0, NULL, NULL);  CLERR(ret)

    clReleaseMemObject(Aprefix);
    clReleaseMemObject(Sprefix);
}



#endif
