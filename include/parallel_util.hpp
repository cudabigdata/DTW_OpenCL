#ifndef PARALLEL_UTIL_CUH
#define PARALLEL_UTIL_CUH

#include "scan.h"
#include "opencl_def.hpp"
#include "CLRadixSort.hpp"


void reduce(cl_mem input, cl_mem output,
            int L) {
    

	cl_kernel reduce_min = SystemCL::Inst().K_reduction_min;
	size_t local_work_size = BLOCKDIM;
	size_t num_blocks = (L -1 )/local_work_size + 1;
	int ret;
	cl_mem d_partial =  clCreateBuffer(SystemCL::Inst().Context(), CL_MEM_READ_WRITE,
			sizeof(float) * (num_blocks + 1), NULL, &ret);  CLERR(ret);
	ret = clFinish(SystemCL::Inst().Queue());
	CLERR(ret)
    size_t global_work_size = num_blocks * local_work_size;
	ret = clSetKernelArg(reduce_min, 0, sizeof(cl_mem), (void *) &input); CLERR(ret)
	ret = clSetKernelArg(reduce_min, 1, sizeof(cl_mem), (void *) &d_partial); CLERR(ret)
	ret = clSetKernelArg(reduce_min, 2, sizeof(int ), (void *) &L); CLERR(ret)
	ret = clSetKernelArg(reduce_min, 3, local_work_size * sizeof(float ), 0 ); CLERR(ret)
	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
			reduce_min, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	CLERR(ret)
	cl_command_queue queue = SystemCL::Inst().Queue();
	ret = clFinish(queue);
	CLERR(ret)

	ret = clSetKernelArg(reduce_min, 0, sizeof(cl_mem), (void *) &d_partial); CLERR(ret)
	ret = clSetKernelArg(reduce_min, 1, sizeof(cl_mem), (void *) &output); CLERR(ret)
	ret = clSetKernelArg(reduce_min, 2, sizeof(int ), (void *) &num_blocks); CLERR(ret)
	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
			reduce_min, 1, NULL, &local_work_size, &local_work_size, 0, NULL, NULL);
	CLERR(ret)
	ret = clFinish(queue);
	CLERR(ret)

	clReleaseMemObject(d_partial);


}


void prefix_sum(cl_mem Input,
                int L) {

    // init entry zero with zero
    double init = 0;
    int ret;
    ret = clEnqueueWriteBuffer(SystemCL::Inst().Queue(), Input, CL_TRUE, 0,
       		 sizeof(double), & init, 0, NULL, NULL);   CLERR(ret)


    Scan::Inst().scan(Input, L);
}

void inc_prefix_sum(cl_mem Input, cl_mem Output, int L) {

	FScan::Inst().scan(Input,Output,  L);
}


void pair_sort(cl_mem Keys, cl_mem  Values, int L) {

    CLRadixSort sort(Keys, Values, L);
    sort.Sort();
    sort.RecupGPU(Keys, Values, L);

}


void threshold(cl_mem Series, int Series_offset, cl_mem Indices, int Indices_offset,  int L, int * newL,
              float bsf) {

    cl_mem Mask = NULL;
    cl_mem Prfx = NULL;
    cl_context context = SystemCL::Inst().Context();
    cl_int ret;
    // device side memory
    Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int)*L, NULL, &ret);    CLERR(ret);
    Prfx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*L, NULL, &ret);    CLERR(ret);

    cl_kernel thresh = SystemCL::Inst().K_thresh;

    ret = clSetKernelArg(thresh, 0, sizeof(cl_mem), (void *) &Series); CLERR(ret)
    ret = clSetKernelArg(thresh, 1, sizeof(int), (void *) &Series_offset); CLERR(ret)
    ret = clSetKernelArg(thresh, 2, sizeof(cl_mem), (void *) &Mask); CLERR(ret)
    ret = clSetKernelArg(thresh, 3, sizeof(int), (void *) &L); CLERR(ret)
    ret = clSetKernelArg(thresh, 4, sizeof(float), (void *) &bsf); CLERR(ret)

    size_t local_work_size = BLOCKDIM;
    size_t global_work_size = local_work_size * SDIV(L, local_work_size);
	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
			thresh, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	CLERR(ret)
	ret = clFinish(SystemCL::Inst().Queue());
	CLERR(ret)

    inc_prefix_sum(Mask, Prfx, L);

	ret = clEnqueueReadBuffer(SystemCL::Inst().Queue(), Prfx, CL_TRUE, (L-1) * sizeof(int),
					sizeof(int), newL, 0, NULL, NULL); CLERR(ret)

	 cl_kernel merge = SystemCL::Inst().K_merge;
	ret = clSetKernelArg(merge, 0, sizeof(cl_mem), (void *) &Mask); CLERR(ret)
	ret = clSetKernelArg(merge, 1, sizeof(cl_mem), (void *) &Prfx); CLERR(ret)
	ret = clSetKernelArg(merge, 2, sizeof(int), (void *) &L); CLERR(ret)

	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
			merge, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	CLERR(ret)
	ret = clFinish(SystemCL::Inst().Queue());
	CLERR(ret)

	 cl_kernel collect = SystemCL::Inst().K_collect;
	ret = clSetKernelArg(collect, 0, sizeof(cl_mem), (void *) &Indices); CLERR(ret)
	ret = clSetKernelArg(collect, 1, sizeof(int), (void *) &Indices_offset); CLERR(ret)
	ret = clSetKernelArg(collect, 2, sizeof(cl_mem), (void *) &Mask); CLERR(ret)
	ret = clSetKernelArg(collect, 3, sizeof(cl_mem), (void *) &Prfx); CLERR(ret)
	ret = clSetKernelArg(collect, 4, sizeof(int), (void *) &L); CLERR(ret)


	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
			collect, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	CLERR(ret)
	ret = clFinish(SystemCL::Inst().Queue());
	CLERR(ret)

	ret = clEnqueueCopyBuffer(SystemCL::Inst().Queue(), Mask, Indices, 0, Indices_offset * sizeof(int), sizeof(int)*L,
			0, NULL, NULL);CLERR(ret)

    clReleaseMemObject(Mask);
    clReleaseMemObject(Prfx);
}

#endif
