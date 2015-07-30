

#ifndef SYSTEMCL_HPP_
#define SYSTEMCL_HPP_

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "opencl_def.hpp"

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>


class SystemCL
{
public:
	cl_kernel K_plainCpy;
	cl_kernel K_squareCpy;
	cl_kernel K_window;
	cl_kernel K_stddev;
	cl_kernel K_set_infty;
	cl_kernel K_register_lb_kim;
	cl_kernel K_range;

	// These kernel for radix sort
	cl_kernel ckTranspose; // transpose the initial list
	cl_kernel ckHistogram;  // compute histograms
	cl_kernel ckScanHistogram; // scan local histogram
	cl_kernel ckPasteHistogram; // paste local histograms
	cl_kernel ckReorder; // final reordering

	cl_kernel K_sparse_block_cdtw;
	cl_kernel K_dense_block_cdtw;
	cl_kernel K_thread_cdtw;

	cl_kernel K_reduction_min;
	cl_kernel K_random_lb_keogh;

	cl_kernel K_thresh;
	cl_kernel K_merge;
	cl_kernel K_collect;
private:

	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_uint num_devices;
	cl_uint num_platforms;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;


    ~SystemCL(){
        cl_int ret = clReleaseProgram(program);

        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
    }
	SystemCL(){


	    platform_id = NULL;
	    device_id = NULL;

	    cl_int ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
	    CLERR(ret);
	    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
	            &device_id, &num_devices);
	    CLERR(ret);
	    // Create an OpenCL context
	    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
	    CLERR(ret);
	    // Create a command queue
	    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	    CLERR(ret);

		std::ifstream kernelFile("opencl_kernel.cl", std::ios::in);
		if (!kernelFile.is_open())
		{
			std::cerr << "Failed to opencl_kernel.cl file for reading: " << std::endl;
			std::exit(0);
		}

		std::ostringstream oss;
		oss << kernelFile.rdbuf();

		std::string srcStdStr = oss.str();
		const char *srcStr = srcStdStr.c_str();
		program = clCreateProgramWithSource(context, 1,
											(const char**)&srcStr,
											NULL, &ret);
		CLERR(ret);
		if (program == NULL)
		{
			std::cerr << "Failed to create CL program from source." << std::endl;
			std::exit(0);
		}

		ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if (ret != CL_SUCCESS)
		{
			// Determine the reason for the error
			char buildLog[16384];
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
								  sizeof(buildLog), buildLog, NULL);

			std::cerr << "Error in kernel: " << std::endl;
			std::cerr << buildLog;
			clReleaseProgram(program);
			std::exit(0);
		}

		// Load all kernel
		K_plainCpy = clCreateKernel(program, "plainCpy", &ret);  CLERR(ret)
		K_squareCpy= clCreateKernel(program, "squareCpy", &ret);  CLERR(ret)
		K_window= clCreateKernel(program, "window", &ret);  CLERR(ret)
		K_stddev = clCreateKernel(program, "stddev", &ret);  CLERR(ret)
		K_set_infty = clCreateKernel(program, "set_infty", &ret);  CLERR(ret)
		K_register_lb_kim = clCreateKernel(program, "register_lb_kim", &ret);  CLERR(ret)
		K_range = clCreateKernel(program, "range", &ret);  CLERR(ret)

		ckHistogram = clCreateKernel(program, "histogram", &ret); CLERR(ret)
		ckScanHistogram = clCreateKernel(program, "scanhistograms", &ret); CLERR(ret)
		ckPasteHistogram = clCreateKernel(program, "pastehistograms", &ret); CLERR(ret)
		ckReorder = clCreateKernel(program, "reorder", &ret); CLERR(ret)
		ckTranspose = clCreateKernel(program, "transpose", &ret); CLERR(ret)


		K_sparse_block_cdtw = clCreateKernel(program, "sparse_block_cdtw", &ret); CLERR(ret)
		K_dense_block_cdtw = clCreateKernel(program, "dense_block_cdtw", &ret); CLERR(ret)
		K_thread_cdtw = clCreateKernel(program, "thread_cdtw", &ret); CLERR(ret)
		K_reduction_min = clCreateKernel(program, "reduction_min", &ret); CLERR(ret)
		K_random_lb_keogh = clCreateKernel(program, "random_lb_keogh", &ret); CLERR(ret)
		K_thresh = clCreateKernel(program, "thresh", &ret); CLERR(ret)

		K_merge = clCreateKernel(program, "merge", &ret); CLERR(ret)
		K_collect = clCreateKernel(program, "collect", &ret); CLERR(ret)
	}

public:

	cl_context & Context(){

		return context;
	}
	cl_command_queue Queue(){
		return command_queue;
	}

	cl_device_id Device(){
		return device_id;
	}

	static SystemCL& Inst()
	{
	  static SystemCL INSTANCE;
	  return INSTANCE;
	}



};


#endif /* SYSTEMCL_HPP_ */
