#ifndef OPENCL_DEF_CUH
#define OPENCL_DEF_CUH

#include <stdio.h>

#define CLERR(err) {  \
    if (err != CL_SUCCESS) { \
        printf("OpenCL error: %d : %s, line %d\n", err, __FILE__, __LINE__); exit(1);}}



#define SDIV(x,y) ((x+y-1)/y)

#define GRIDDIM (1024)
#define BLOCKDIM (128)


#endif
