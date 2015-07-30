#ifndef BOUNDS_CUH
#define BOUNDS_CUH

#include<list>             // envelope


void envelope(float* series, int W,
              float* L, float* U, int M) {
    
    // Daniel Lemire's windowed min-max algorithm in O(3n):
    std::list<int> u = std::list<int>();
    std::list<int> l = std::list<int>();

    u.push_back(0);
    l.push_back(0);

    for (int i = 1; i < M; ++i) {

        if (i > W) {

            U[i-W-1] = series[u.front()];
            L[i-W-1] = series[l.front()];
        }
        
        if (series[i] > series[i-1]) {
            
            u.pop_back();
            while (!u.empty() && series[i] > series[u.back()])
                u.pop_back();
        } else {

            l.pop_back();
            while (!l.empty() && series[i] < series[l.back()])
                l.pop_back();
        }
        
        u.push_back(i);
        l.push_back(i);
        
        if (i == 2*W+1+u.front())
            u.pop_front();
        else if (i == 2*W+1+l.front())
            l.pop_front();
    }

    for (int i = M; i < M+W+1; ++i) {

        U[i-W-1] = series[u.front()];
        L[i-W-1] = series[l.front()];

        if (i-u.front() >= 2*W+1)
            u.pop_front();

        if (i-l.front() >= 2*W+1)
            l.pop_front();
    }
}


void lb_kim(cl_mem Subject,cl_mem AvgS,cl_mem StdS,cl_mem Lb_kim,  int M, int N) {

    if (M < 11) {
        std::cout << "ERROR: LB_Kim is a hardcoded 10 point unconstrained DTW."
                  << "Chose a query length of at least 11!" << std::endl;
        return;
    }
    cl_int ret;
    
    cl_kernel register_lb_kim = SystemCL::Inst().K_register_lb_kim;

	ret = clSetKernelArg(register_lb_kim, 0, sizeof(cl_mem), (void *) &Subject); CLERR(ret)
	ret = clSetKernelArg(register_lb_kim, 1, sizeof(cl_mem), (void *) &AvgS); CLERR(ret)
	ret = clSetKernelArg(register_lb_kim, 2, sizeof(cl_mem), (void *) &StdS); CLERR(ret)
	ret = clSetKernelArg(register_lb_kim, 3, sizeof(cl_mem), (void *) &Lb_kim); CLERR(ret)
	ret = clSetKernelArg(register_lb_kim, 4, sizeof(int), (void *) &M); CLERR(ret)
	ret = clSetKernelArg(register_lb_kim, 5, sizeof(int), (void *) &N); CLERR(ret)
	ret = clSetKernelArg(register_lb_kim, 6, sizeof(cl_mem), (void *) &Czquery); CLERR(ret)

	size_t global_work_size = SDIV(N-M+1, BLOCKDIM) * BLOCKDIM;

	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
			register_lb_kim, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
	CLERR(ret)

}



void lb_keogh(cl_mem Subject,cl_mem AvgS,cl_mem StdS, cl_mem Lb_keogh, int Lb_keogh_offset,
              int M, cl_mem Indices, int Indices_offset, int indices_l) {


    cl_kernel random_lb_keogh = SystemCL::Inst().K_random_lb_keogh;
    int ret;
    ret = clSetKernelArg(random_lb_keogh, 0, sizeof(cl_mem), (void *) &Subject); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 1, sizeof(cl_mem), (void *) &AvgS); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 2, sizeof(cl_mem), (void *) &StdS); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 3, sizeof(cl_mem), (void *) &Lb_keogh); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 4, sizeof(int), (void *) &Lb_keogh_offset); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 5, sizeof(int), (void *) &M); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 6, sizeof(cl_mem), (void *) &Indices); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 7, sizeof(int), (void *) &Indices_offset); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 8, sizeof(int), (void *) &indices_l); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 9, sizeof(cl_mem), (void *) &Czlower); CLERR(ret)
    ret = clSetKernelArg(random_lb_keogh, 10, sizeof(cl_mem), (void *) &Czupper); CLERR(ret)

	size_t local_work_size = 512;
	int griddim = SDIV(indices_l, local_work_size);
	size_t global_work_size = griddim * local_work_size;

	ret = clEnqueueNDRangeKernel(SystemCL::Inst().Queue(),
			random_lb_keogh, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
	CLERR(ret)

	ret = clFinish(SystemCL::Inst().Queue());
	CLERR(ret)
}

#endif
