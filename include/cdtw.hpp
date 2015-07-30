#ifndef CDTW_CUH
#define CDTW_CUH

//#include <omp.h>                // omp pragmas
#include "opencl_def.hpp"         // CUERR makro
#include "stats.hpp"            // statistics of time series
#include "SystemCL.hpp"
#include <algorithm>
#define MAXQUERYSIZE (4096)

cl_mem Czlower;                       // constant memory
cl_mem Czupper;
cl_mem Czquery;

// insert code for lower bounds here
#include "bounds.hpp"

void gpu_cdtw(cl_mem Subject,cl_mem AvgS, cl_mem StdS,
              cl_mem Cdtw, int M, cl_mem Indices, int indices_l,
              int W, bool blockdtw = true){

    if (blockdtw) {
        int lane = W+2;
        int size = 128/lane;

        if ((3*lane+M)*sizeof(float) > 48*1024 or lane > 1024) {
            std::cout << "ERROR: Not enough shared memory or threads present: "
                      << "please decrease query or window size. "
                      << "Alternatively use threaded DTW (blockdtw=false)."
                      << std::endl;
            return;
        }

        if (size == 0) {

        	cl_kernel sparse_block_cdtw = SystemCL::Inst().K_sparse_block_cdtw;
        	int ret;
        	ret = clSetKernelArg(sparse_block_cdtw, 0, sizeof(cl_mem), (void *) &Subject); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 1, sizeof(cl_mem), (void *) &AvgS); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 2, sizeof(cl_mem), (void *) &StdS); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 3, sizeof(cl_mem), (void *) &Cdtw); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 4, sizeof(int), (void *) &M); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 5, sizeof(cl_mem), (void *) &Indices); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 6, sizeof(int), (void *) &indices_l); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 7, sizeof(int), (void *) &W); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 8, (3*lane+M)*sizeof(float), NULL); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 9, sizeof(cl_mem), (void *) &Subject); CLERR(ret)
        	ret = clSetKernelArg(sparse_block_cdtw, 10, sizeof(cl_mem), (void *) &Czquery); CLERR(ret)
        	size_t global_work_size = indices_l * lane;
        	size_t local_work_size = lane;
        	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
        			sparse_block_cdtw, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        	CLERR(ret)


        } else {
            int blockdim = size*lane;
            int griddim = SDIV(indices_l, size);

            cl_kernel dense_block_cdtw = SystemCL::Inst().K_dense_block_cdtw;
			int ret;
			ret = clSetKernelArg(dense_block_cdtw, 0, sizeof(cl_mem), (void *) &Subject); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 1, sizeof(cl_mem), (void *) &AvgS); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 2, sizeof(cl_mem), (void *) &StdS); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 3, sizeof(cl_mem), (void *) &Cdtw); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 4, sizeof(int), (void *) &M); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 5, sizeof(cl_mem), (void *) &Indices); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 6, sizeof(int), (void *) &indices_l); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 7, sizeof(int), (void *) &W); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 8,  (M+size*3*lane)*sizeof(float), NULL); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 9, sizeof(cl_mem), (void *) &Subject); CLERR(ret)
			ret = clSetKernelArg(dense_block_cdtw, 10, sizeof(cl_mem), (void *) &Czquery); CLERR(ret)
			size_t global_work_size = SDIV(indices_l, size) * size * lane;
			size_t local_work_size = size*lane;

			ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
					dense_block_cdtw, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			CLERR(ret)
			ret = clFinish(SystemCL::Inst().Queue());
			CLERR(ret)

        }

    } else {
        int blockdim = 64;
        int griddim = SDIV(indices_l, blockdim);

        cl_kernel thread_cdtw = SystemCL::Inst().K_thread_cdtw;
		int ret;
		ret = clSetKernelArg(thread_cdtw, 0, sizeof(cl_mem), (void *) &Subject); CLERR(ret)
		ret = clSetKernelArg(thread_cdtw, 1, sizeof(cl_mem), (void *) &AvgS); CLERR(ret)
		ret = clSetKernelArg(thread_cdtw, 2, sizeof(cl_mem), (void *) &StdS); CLERR(ret)
		ret = clSetKernelArg(thread_cdtw, 3, sizeof(cl_mem), (void *) &Cdtw); CLERR(ret)
		ret = clSetKernelArg(thread_cdtw, 4, sizeof(int), (void *) &M); CLERR(ret)
		ret = clSetKernelArg(thread_cdtw, 5, sizeof(cl_mem), (void *) &Indices); CLERR(ret)
		ret = clSetKernelArg(thread_cdtw, 6, sizeof(int), (void *) &indices_l); CLERR(ret)
		ret = clSetKernelArg(thread_cdtw, 7, sizeof(int), (void *) &W); CLERR(ret)
		ret = clSetKernelArg(thread_cdtw, 8, sizeof(cl_mem), (void *) &Subject); CLERR(ret)
		ret = clSetKernelArg(thread_cdtw, 9, sizeof(cl_mem), (void *) &Czquery); CLERR(ret)
		size_t global_work_size = SDIV(indices_l, blockdim) * 64;
		size_t local_work_size = 64;
		ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
				thread_cdtw, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
		CLERR(ret)
		ret = clFinish(SystemCL::Inst().Queue());
		CLERR(ret)

    }
}

///////////////////////////////////////////////////////////////////////////////
// GPU lower bounded condstrained DTW
///////////////////////////////////////////////////////////////////////////////


void report_result(cl_mem Cdtw, cl_mem Indices, int upper) {

    pair_sort(Cdtw, Indices, upper);
    float bsf = INFINITY;
    int bsf_index = -1;
    
    int ret = clEnqueueReadBuffer(SystemCL::Inst().Queue(), Cdtw, CL_TRUE, 0,
             		sizeof(float), &bsf, 0, NULL, NULL); CLERR(ret)

    ret = clEnqueueReadBuffer(SystemCL::Inst().Queue(), Indices, CL_TRUE, 0,
             		sizeof(int), &bsf_index, 0, NULL, NULL); CLERR(ret)
    std:: cout << "Location: " << bsf_index << std::endl;
    std:: cout << "Distance: " <<  sqrt(bsf)  << std::endl;
}

int get_result(cl_mem Cdtw, cl_mem Indices, int upper) {

    pair_sort(Cdtw, Indices, upper);
    int bsf_index = -1;

    cl_int ret = clEnqueueReadBuffer(SystemCL::Inst().Queue(), Indices, CL_TRUE, 0,
             		sizeof(int), &bsf_index, 0, NULL, NULL); CLERR(ret)

    return bsf_index;
}





int prune_cdtw(cl_mem Subject, cl_mem AvgS, cl_mem StdS, int M, int N, int W ) {


	int matched = -1;

    cl_mem Lb_kim = NULL;
    cl_mem Lb_keogh = NULL;
    cl_mem Cdtw = NULL;
    cl_mem Best_cdtw = NULL;

    float max_kim = INFINITY, best_cdtw = INFINITY, tmp = INFINITY;
    
    cl_mem Indices = NULL;
    
    cl_int ret;
    cl_context context = SystemCL::Inst().Context();

    Best_cdtw = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &ret);  CLERR(ret);
    Lb_kim = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(N-M+1), NULL, &ret);  CLERR(ret);
    Lb_keogh = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(N-M+1), NULL, &ret);  CLERR(ret);
    Cdtw = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(N-M+1), NULL, &ret);  CLERR(ret);
    Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(N-M+1), NULL, &ret);  CLERR(ret);
    
    // init Cdtw with infinity
    cl_kernel set_infty = SystemCL::Inst().K_set_infty;
    int L = N-M+1;
    float Infinity = INFINITY;
	ret = clSetKernelArg(set_infty, 0, sizeof(cl_mem), (void *) &Cdtw);
	ret = clSetKernelArg(set_infty, 1, sizeof(int), (void *) &L);
	ret = clSetKernelArg(set_infty, 2, sizeof(float), (void *) &Infinity);

	size_t global_work_size = SDIV(N-M+1, BLOCKDIM) * BLOCKDIM;
	size_t local_work_size = BLOCKDIM;
	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
			set_infty, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	CLERR(ret)

    
    // calculate LB_Kim for all entries and sort indices
    lb_kim(Subject, AvgS, StdS, Lb_kim, M, N );

	cl_kernel range = SystemCL::Inst().K_range;
	ret = clSetKernelArg(range, 0, sizeof(cl_mem), (void *) &Indices);
	ret = clSetKernelArg(range, 1, sizeof(int), (void *) &L);

	global_work_size = GRIDDIM * BLOCKDIM;
	local_work_size = BLOCKDIM;
	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
			range, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	CLERR(ret)


    pair_sort(Lb_kim, Indices, N-M+1);


    int chunk = std::max(std::min(1<<10, N-M+1), 2*M);
    int lower = 0, upper = chunk;
    
    gpu_cdtw(Subject, AvgS, StdS, Cdtw, M, Indices, chunk, W,
                        true);

    reduce(Cdtw, Best_cdtw, chunk);

    ret = clEnqueueReadBuffer(SystemCL::Inst().Queue(), Best_cdtw, CL_TRUE, 0,
             		sizeof(float), &best_cdtw, 0, NULL, NULL); CLERR(ret)


	ret = clEnqueueReadBuffer(SystemCL::Inst().Queue(), Lb_kim, CL_TRUE, (chunk -1) * sizeof(float),
					sizeof(float), &max_kim, 0, NULL, NULL); CLERR(ret)


    if (best_cdtw < max_kim) {
    	matched = get_result(Cdtw, Indices, upper);
    } else {
        for (lower = upper; lower < N-M+1; lower += chunk) {
            
            // update upper index and double chunk size
            chunk = min(2*chunk, 1<<16);
            upper = min(upper+chunk, N-M+1);
            
            // calculate Lb_Keogh on current chunk
            lb_keogh(Subject, AvgS, StdS, Lb_keogh, lower,
                                M, Indices, lower, upper-lower);
            
            int length;
            threshold(Lb_keogh,lower, Indices,lower, upper-lower,
                      &length, best_cdtw);
            
            if (length == 0)
                continue;
            

            gpu_cdtw(Subject, AvgS, StdS, Cdtw, M,
                                Indices, length, W, true);

            reduce(Cdtw, Best_cdtw, length);
    
			ret = clEnqueueReadBuffer(SystemCL::Inst().Queue(), Best_cdtw, CL_TRUE, 0,
							sizeof(float), &tmp, 0, NULL, NULL); CLERR(ret)

			ret = clEnqueueReadBuffer(SystemCL::Inst().Queue(), Lb_kim, CL_TRUE, sizeof(float) * (upper-1),
									sizeof(float), &max_kim, 0, NULL, NULL); CLERR(ret)
            
            best_cdtw = std::min(tmp, best_cdtw);

            if (best_cdtw < max_kim) {
                break;
            }
        }
    }
    
	matched = get_result(Cdtw, Indices, upper);


    clReleaseMemObject(Best_cdtw);
    clReleaseMemObject(Lb_kim);
    clReleaseMemObject(Lb_keogh);
    clReleaseMemObject(Cdtw);
    clReleaseMemObject(Indices);

    return matched;

}

#endif


