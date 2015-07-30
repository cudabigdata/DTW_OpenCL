#ifndef INCLUDE_SCAN_H_
#define INCLUDE_SCAN_H_

#include "SystemCL.hpp"
#include <cmath>
class Scan {
private:

	cl_kernel scan_pow2;
	cl_kernel scan_pad_to_pow2;
	cl_kernel scan_subarrays;
	cl_kernel scan_inc_subarrays;
	size_t wx; // workgroup size
	int m;     // length of each subarray ( = wx*2 )

	void recursive_scan(cl_mem d_data, int n) {
		int k = (int) ceil((float) n / (float) m);
		//size of each subarray stored in local memory
		size_t bufsize = sizeof(double) * m;
		cl_context context = SystemCL::Inst().Context();
		cl_command_queue queue = SystemCL::Inst().Queue();
		cl_int ret;

		if (k == 1) {
			ret = clSetKernelArg(scan_pad_to_pow2, 0, sizeof(cl_mem),
					(void *) &d_data);
			CLERR(ret)
			ret = clSetKernelArg(scan_pad_to_pow2, 1, bufsize, NULL);
			CLERR(ret)
			ret = clSetKernelArg(scan_pad_to_pow2, 2, sizeof(int), (void *) &n);
			CLERR(ret)

			ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
					scan_pad_to_pow2, 1, NULL, &wx, &wx, 0, NULL, NULL);
			CLERR(ret)

		} else {

			size_t gx = k * wx;
			cl_mem d_partial = clCreateBuffer(context, CL_MEM_READ_WRITE,
					sizeof(double) * k, NULL, &ret);
			CLERR(ret);

			ret = clSetKernelArg(scan_subarrays, 0, sizeof(cl_mem),
					(void *) &d_data);
			CLERR(ret)
			ret = clSetKernelArg(scan_subarrays, 1, bufsize, NULL);
			CLERR(ret)
			ret = clSetKernelArg(scan_subarrays, 2, sizeof(cl_mem),
					(void *) &d_partial);
			CLERR(ret)
			ret = clSetKernelArg(scan_subarrays, 3, sizeof(int), (void *) &n);
			CLERR(ret)

			ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
					scan_subarrays, 1, NULL, &gx, &wx, 0, NULL, NULL);
			CLERR(ret)
			clFinish(SystemCL::Inst().Queue());

			recursive_scan(d_partial, k);

			ret = clSetKernelArg(scan_inc_subarrays, 0, sizeof(cl_mem),
					(void *) &d_data);
			CLERR(ret)
			ret = clSetKernelArg(scan_inc_subarrays, 1, bufsize, NULL);
			CLERR(ret)
			ret = clSetKernelArg(scan_inc_subarrays, 2, sizeof(cl_mem),
					(void *) &d_partial);
			CLERR(ret)
			ret = clSetKernelArg(scan_inc_subarrays, 3, sizeof(int),
					(void *) &n);
			CLERR(ret)

			ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
					scan_inc_subarrays, 1, NULL, &gx, &wx, 0, NULL, NULL);
			CLERR(ret)


			clReleaseMemObject(d_partial);


		}
	}

public:
	Scan(size_t _wx = BLOCKDIM) {
		wx = _wx;
		m = wx * 2;

		std::ifstream kernelFile("scan.cl", std::ios::in);
		if (!kernelFile.is_open()) {
			std::cerr << "Failed to scan.cl file for reading: " << std::endl;
			std::exit(0);
		}

		std::ostringstream oss;
		oss << kernelFile.rdbuf();

		std::string srcStdStr = oss.str();
		const char *srcStr = srcStdStr.c_str();

		cl_context context = SystemCL::Inst().Context();
		cl_int ret;

		cl_program program = clCreateProgramWithSource(context, 1,
				(const char**) &srcStr,
				NULL, &ret);
		CLERR(ret);
		if (program == NULL) {
			std::cerr << "Failed to create CL program from source."
					<< std::endl;
			std::exit(0);
		}

		ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if (ret != CL_SUCCESS) {
			// Determine the reason for the error
			char buildLog[16384];
			clGetProgramBuildInfo(program, SystemCL::Inst().Device(),
					CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

			std::cerr << "Error in kernel: " << std::endl;
			std::cerr << buildLog;
			clReleaseProgram(program);
			std::exit(0);
		}

		scan_pow2 = clCreateKernel(program, "scan_pow2_wrapper", &ret);
		CLERR(ret)
		scan_pad_to_pow2 = clCreateKernel(program, "scan_pad_to_pow2", &ret);
		CLERR(ret)
		scan_subarrays = clCreateKernel(program, "scan_subarrays", &ret);
		CLERR(ret)
		scan_inc_subarrays = clCreateKernel(program, "scan_inc_subarrays",
				&ret);
		CLERR(ret)

	}

	void scan(cl_mem data, int n) {
		int k = (int) ceil((float) n / (float) m);
		cl_int ret;
		cl_context context = SystemCL::Inst().Context();
		cl_command_queue queue = SystemCL::Inst().Queue();
		cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE,
				sizeof(double) * (k * m), NULL, &ret);
		CLERR(ret);

		ret = clEnqueueCopyBuffer(queue, data, d_data, 0, 0, sizeof(double) * n,
				0, NULL, NULL);
		CLERR(ret);

		recursive_scan(d_data, n);

		ret = clEnqueueCopyBuffer(queue, d_data, data, 0, 0, sizeof(double) * n,
				0, NULL, NULL);
		CLERR(ret);
		clReleaseMemObject(d_data);
	}

	void scan(cl_mem data, cl_mem out,  int n) {
		int k = (int) ceil((float) n / (float) m);
		cl_int ret;
		cl_context context = SystemCL::Inst().Context();
		cl_command_queue queue = SystemCL::Inst().Queue();
		cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE,
				sizeof(double) * (k * m), NULL, &ret);
		CLERR(ret);

		ret = clEnqueueCopyBuffer(queue, data, d_data, 0, 0, sizeof(double) * n,
				0, NULL, NULL);
		CLERR(ret);

		recursive_scan(d_data, n);

		ret = clEnqueueCopyBuffer(queue, d_data, out, 0, 0, sizeof(double) * n,
				0, NULL, NULL);
		CLERR(ret);
		clReleaseMemObject(d_data);
	}

	static Scan& Inst() {
		static Scan INSTANCE;
		return INSTANCE;
	}
	;
};



class FScan {
private:

	cl_kernel scan_pow2;
	cl_kernel scan_pad_to_pow2;
	cl_kernel scan_subarrays;
	cl_kernel scan_inc_subarrays;
	size_t wx; // workgroup size
	int m;     // length of each subarray ( = wx*2 )

	void recursive_scan(cl_mem d_data, int n) {
		int k = (int) ceil((float) n / (float) m);
		//size of each subarray stored in local memory
		size_t bufsize = sizeof(float) * m;
		cl_context context = SystemCL::Inst().Context();
		cl_command_queue queue = SystemCL::Inst().Queue();
		cl_int ret;

		if (k == 1) {
			ret = clSetKernelArg(scan_pad_to_pow2, 0, sizeof(cl_mem),
					(void *) &d_data);
			CLERR(ret)
			ret = clSetKernelArg(scan_pad_to_pow2, 1, bufsize, NULL);
			CLERR(ret)
			ret = clSetKernelArg(scan_pad_to_pow2, 2, sizeof(int), (void *) &n);
			CLERR(ret)

			ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
					scan_pad_to_pow2, 1, NULL, &wx, &wx, 0, NULL, NULL);
			CLERR(ret)

		} else {

			size_t gx = k * wx;
			cl_mem d_partial = clCreateBuffer(context, CL_MEM_READ_WRITE,
					sizeof(float) * k, NULL, &ret);
			CLERR(ret);

			ret = clSetKernelArg(scan_subarrays, 0, sizeof(cl_mem),
					(void *) &d_data);
			CLERR(ret)
			ret = clSetKernelArg(scan_subarrays, 1, bufsize, NULL);
			CLERR(ret)
			ret = clSetKernelArg(scan_subarrays, 2, sizeof(cl_mem),
					(void *) &d_partial);
			CLERR(ret)
			ret = clSetKernelArg(scan_subarrays, 3, sizeof(int), (void *) &n);
			CLERR(ret)

			ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
					scan_subarrays, 1, NULL, &gx, &wx, 0, NULL, NULL);
			CLERR(ret)
			clFinish(SystemCL::Inst().Queue());

			recursive_scan(d_partial, k);

			ret = clSetKernelArg(scan_inc_subarrays, 0, sizeof(cl_mem),
					(void *) &d_data);
			CLERR(ret)
			ret = clSetKernelArg(scan_inc_subarrays, 1, bufsize, NULL);
			CLERR(ret)
			ret = clSetKernelArg(scan_inc_subarrays, 2, sizeof(cl_mem),
					(void *) &d_partial);
			CLERR(ret)
			ret = clSetKernelArg(scan_inc_subarrays, 3, sizeof(int),
					(void *) &n);
			CLERR(ret)

			ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
					scan_inc_subarrays, 1, NULL, &gx, &wx, 0, NULL, NULL);
			CLERR(ret)


			clReleaseMemObject(d_partial);


		}
	}

public:
	FScan(size_t _wx = BLOCKDIM) {
		wx = _wx;
		m = wx * 2;

		std::ifstream kernelFile("scan.cl", std::ios::in);
		if (!kernelFile.is_open()) {
			std::cerr << "Failed to scan.cl file for reading: " << std::endl;
			std::exit(0);
		}

		std::ostringstream oss;
		oss << kernelFile.rdbuf();

		std::string srcStdStr = oss.str();
		const char *srcStr = srcStdStr.c_str();

		cl_context context = SystemCL::Inst().Context();
		cl_int ret;

		cl_program program = clCreateProgramWithSource(context, 1,
				(const char**) &srcStr,
				NULL, &ret);
		CLERR(ret);
		if (program == NULL) {
			std::cerr << "Failed to create CL program from source."
					<< std::endl;
			std::exit(0);
		}

		ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if (ret != CL_SUCCESS) {
			// Determine the reason for the error
			char buildLog[16384];
			clGetProgramBuildInfo(program, SystemCL::Inst().Device(),
					CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

			std::cerr << "Error in kernel: " << std::endl;
			std::cerr << buildLog;
			clReleaseProgram(program);
			std::exit(0);
		}

		scan_pow2 = clCreateKernel(program, "fscan_pow2_wrapper", &ret);
		CLERR(ret)
		scan_pad_to_pow2 = clCreateKernel(program, "fscan_pad_to_pow2", &ret);
		CLERR(ret)
		scan_subarrays = clCreateKernel(program, "fscan_subarrays", &ret);
		CLERR(ret)
		scan_inc_subarrays = clCreateKernel(program, "fscan_inc_subarrays",
				&ret);
		CLERR(ret)

	}

	void scan(cl_mem data, int n) {
		int k = (int) ceil((float) n / (float) m);
		cl_int ret;
		cl_context context = SystemCL::Inst().Context();
		cl_command_queue queue = SystemCL::Inst().Queue();
		cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE,
				sizeof(float) * (k * m), NULL, &ret);
		CLERR(ret);

		ret = clEnqueueCopyBuffer(queue, data, d_data, 0, 0, sizeof(float) * n,
				0, NULL, NULL);
		CLERR(ret);

		recursive_scan(d_data, n);

		ret = clEnqueueCopyBuffer(queue, d_data, data, 0, 0, sizeof(float) * n,
				0, NULL, NULL);
		CLERR(ret);
		clReleaseMemObject(d_data);
	}

	void scan(cl_mem data, cl_mem out,  int n) {
		int k = (int) ceil((float) n / (float) m);
		cl_int ret;
		cl_context context = SystemCL::Inst().Context();
		cl_command_queue queue = SystemCL::Inst().Queue();
		cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE,
				sizeof(float) * (k * m), NULL, &ret);
		CLERR(ret);

		ret = clEnqueueCopyBuffer(queue, data, d_data, 0, 0, sizeof(float) * n,
				0, NULL, NULL);
		CLERR(ret);

		recursive_scan(d_data, n);

		ret = clEnqueueCopyBuffer(queue, d_data, out, 0, 0, sizeof(float) * n,
				0, NULL, NULL);
		CLERR(ret);
		clReleaseMemObject(d_data);
	}

	static FScan& Inst() {
		static FScan INSTANCE;
		return INSTANCE;
	}
	;
};


#endif /* INCLUDE_SCAN_H_ */
