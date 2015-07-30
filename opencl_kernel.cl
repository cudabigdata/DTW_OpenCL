#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void plainCpy(__global float * Input, __global double * Output, int L) {

    const int tid = get_local_size(0) * get_group_id(0)  + get_local_id(0);
    int step = get_num_groups(0) * get_local_size(0);
    for (int i = tid; i < L; i += step )
        Output[i] = Input[i];
}
__kernel void squareCpy(__global float * Input, __global double * Output, int L) {

    const int tid = get_local_size(0) * get_group_id(0)  + get_local_id(0);
    int step = get_num_groups(0) * get_local_size(0);
    for (int i = tid; i < L; i += step ){
	double value = Input[i];
        Output[i] = value*value;
    }
       
}

__kernel void window(__global double * Input, __global float * Output, int L, int W) {

    const int tid =  get_local_size(0) * get_group_id(0)  + get_local_id(0);
    int step = get_num_groups(0) * get_local_size(0);
    for (int i = tid; i < L-W+1; i += step)
        Output[i] = (Input[i+W]-Input[i]) / W;
}


__kernel void stddev(__global float * X, __global float * X2,
                  __global float * Output, int L, int W) {

    const int tid = get_local_size(0) * get_group_id(0)  + get_local_id(0);
    int step = get_num_groups(0) * get_local_size(0);
    for (int i = tid; i < L-W+1; i += step) {
        float mu = X[i];
        mu = X2[i] - mu*mu;
        Output[i] = mu > 0 ? sqrt(mu) : 1;
    }
}

__kernel void set_infty(__global float * Series, int L , float inf) {

    int thid = get_local_size(0) * get_group_id(0)  + get_local_id(0);

    if (thid < L)
        Series[thid] = inf;
}

#define euclidean true
#define lp(x,y) ((x) ? (y)*(y) : (y) < 0 ? -(y) : (y))


__kernel void register_lb_kim(__global float * Subject, __global float * AvgS, __global float * StdS, 
                   __global float * Lb_kim, int M, int N, __constant float * Czquery) {

    int indx = get_local_size(0) * get_group_id(0)  + get_local_id(0);
    
    // registers for 10-point DTW
    float q0, q1, q2, q3, q4, s0, s1, s2, s3, s4, mem, 
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
    
    if (indx < N-M+1) {
        

        // get statistics for given index
        float avg = AvgS[indx];
        float std = StdS[indx];
        
        // read query
        q0 = Czquery[0];
        q1 = Czquery[1];
        q2 = Czquery[2];
        q3 = Czquery[3];
        q4 = Czquery[4];
        
        // read subject and z-normalize it
        s0 = (Subject[indx]-avg)/std;
        s1 = (Subject[indx+1]-avg)/std;
        s2 = (Subject[indx+2]-avg)/std;
        s3 = (Subject[indx+3]-avg)/std;
        s4 = (Subject[indx+4]-avg)/std;

        // relax the first row
        p0 = lp(euclidean, q0-s0);
        p1 = lp(euclidean, q0-s1) + p0;
        p2 = lp(euclidean, q0-s2) + p1;
        p3 = lp(euclidean, q0-s3) + p2;
        p4 = lp(euclidean, q0-s4) + p3;
        mem = p4;
        
        // relax the second row
        p5 = lp(euclidean, q1-s0) + p0;
        p6 = lp(euclidean, q1-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q1-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q1-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q1-s4) + min(p8, min(p3, p4));
        mem = min(mem, p9);
        
        // relax the third row
        p0 = lp(euclidean, q2-s0) + p5;
        p1 = lp(euclidean, q2-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q2-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q2-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q2-s4) + min(p3, min(p8, p9));
        mem = min(mem, p4);
        
        // relax the fourth row
        p5 = lp(euclidean, q3-s0) + p0;
        p6 = lp(euclidean, q3-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q3-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q3-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q3-s4) + min(p8, min(p3, p4));
        mem = min(mem, p9);
        
        // relax the fith row
        p0 = lp(euclidean, q4-s0) + p5;
        p1 = lp(euclidean, q4-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q4-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q4-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q4-s4) + min(p3, min(p8, p9));
        mem = min(mem, min(p0, min(p1, min(p2, min(p3, p4)))));
        
        // now do the same for the end of the window
        
        // read query
        q0 = Czquery[M-5];
        q1 = Czquery[M-4];
        q2 = Czquery[M-3];
        q3 = Czquery[M-2];
        q4 = Czquery[M-1];
        
        // read subject and z-normalize it
        s0 = (Subject[indx+M-5]-avg)/std;
        s1 = (Subject[indx+M-4]-avg)/std;
        s2 = (Subject[indx+M-3]-avg)/std;
        s3 = (Subject[indx+M-2]-avg)/std;
        s4 = (Subject[indx+M-1]-avg)/std;

        // relax the first row
        p0 = lp(euclidean, q0-s0);
        p1 = lp(euclidean, q0-s1);
        p2 = lp(euclidean, q0-s2);
        p3 = lp(euclidean, q0-s3);
        p4 = lp(euclidean, q0-s4);
        
        // relax the second row
        p5 = lp(euclidean, q1-s0);
        p6 = lp(euclidean, q1-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q1-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q1-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q1-s4) + min(p8, min(p3, p4));
        
        // relax the third row
        p0 = lp(euclidean, q2-s0);
        p1 = lp(euclidean, q2-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q2-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q2-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q2-s4) + min(p3, min(p8, p9));
        
        // relax the fourth row
        p5 = lp(euclidean, q3-s0);
        p6 = lp(euclidean, q3-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q3-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q3-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q3-s4) + min(p8, min(p3, p4));
       
        // relax the fith row
        p0 = lp(euclidean, q4-s0);
        p1 = lp(euclidean, q4-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q4-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q4-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q4-s4) + min(p3, min(p8, p9));
                
        Lb_kim[indx] = mem+p4;
    }
}


__kernel void range(__global int * Indices, int L) {

    const int thid = get_local_size(0) * get_group_id(0)  + get_local_id(0);
    int step = get_num_groups(0) * get_local_size(0);

    for (int i = thid; i < L; i += step)
        Indices[i] = i;
}


// linear memory constrained block dtw (one dtw per block)
__kernel void sparse_block_cdtw(__global float *  Subject, __global float * AvgS, __global float * StdS, 
                       __global float * Cdtw, int M, __global int * Indices, 
                       int indices_l, int W, __local float * cache, __global float * Tsubject, __global float * Czquery){

    int thid =  get_local_id(0) ;
    int seid = get_group_id(0) ; 

    // retrieve actual index and lane length
    int indx = Indices[seid];
    int lane = W+2;
    int cntr = (W+1)/2;

    __local float * sh_Penalty = cache;
    __local float * sh_Query = cache+3*lane;
    
    if (seid < indices_l) {
        
        // retrieve statistics of subject subsequence
        float avg = AvgS[indx];
        float std = StdS[indx];
        
        // initialize penalty matrix
        for (int m = thid; m < 3*lane; m += get_local_size(0) )
            sh_Penalty[m] = INFINITY;
        sh_Penalty[cntr] = 0;
        
        // initialize shared memory for query
        for (int m =  thid; m < M; m += get_local_size(0) )
            sh_Query[m] = Czquery[m];

        barrier(CLK_LOCAL_MEM_FENCE);

        // initialize diagonal pattern
        int p = W & 1;
        int q = (p == 0);
        
        // k % 2 and k / 2
        int km2 = 1, kd2 = 0;
        
        // row indices
        int trg_row = 1, pr1_row = 0, pr2_row = 2;
        
        // relax diagonal pattern
        for (int k = 2; k < 2*M+1; ++k) {

            // base index of row and associated shift
            kd2 += km2;
            km2 = (km2 == 0);
            int b_i = cntr+kd2+q*km2, b_j = -cntr+kd2+p*km2, shift = (p^km2);
            
            // cyclic indexing of rows
            pr1_row = trg_row;
            pr2_row = trg_row == 0 ? 2 : trg_row - 1;
            trg_row = trg_row == 2 ? 0 : trg_row + 1;
            
            for (int l = thid; l < lane-1; l += get_local_size(0)) {
                
                // potential index of node (if valid)
                int i = b_i-l;
                int j = b_j+l;
                
                // check if in valid relaxation area
                bool inside = 0 < i && i < M+1 && 0 < j && j < M+1 &&
                              shift <= l && l < lane-1;
                
                float value = INFINITY;
                
                if (inside) {
                    value = (Tsubject[indx+j-1]-avg)/std 
                          - sh_Query[i-1];
                    
                    // Euclidean or Manhattan distance?
                    value = euclidean ? value*value :
                            value < 0 ? -value : value;

                    // get mininum incoming edge and relax
                    value += min(sh_Penalty[pr2_row*lane+l],
                             min(sh_Penalty[pr1_row*lane+l-shift],
                                 sh_Penalty[pr1_row*lane+l-shift+1]));
                }
                
                // write value to penalty matrix
                sh_Penalty[trg_row*lane+l] = value;
            }
            // sync threads after each row
             barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // write result down
        Cdtw[seid] = sh_Penalty[((2*M) % 3)*lane+cntr];
    }
}

// linear memory constrained block dtw (many dtws per block)
__kernel void dense_block_cdtw(__global float * Subject, __global float * AvgS, __global float * StdS, 
                      __global float * Cdtw, int M, __global int * Indices, 
                      int indices_l, int W, __local float * cache,__global float * Tsubject, __global float * Czquery){

    // lane length and center node
    const int lane = W+2;
    const int cntr = (W+1)/2;
    
    // number of dtws per block, slot id of dtw and global id
    const int size = get_local_size(0) /lane;
    const int slid = get_local_id(0)/lane;
    const int nidx = get_local_id(0) - slid*lane;
    const int seid = size* get_group_id(0)  + slid;

    // set memory location in shared memory
    __local float * sh_Query = cache;
    __local float * sh_Penalty = sh_Query+M+slid*(3*lane);

    // initialize shared memory for query
    for (int m =  get_local_id(0); m < M; m += get_local_size(0) )
        sh_Query[m] = Czquery[m];

    // initialize penalty matrix
    for (int m = get_local_id(0); m < 3*lane*size; m += get_local_size(0) )
        cache[M+m] = INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);
   
    // set upper left cell to zero
    if (nidx == 0)
        sh_Penalty[cntr] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (seid < indices_l) {
        
        // retrieve statistics of subject subsequence
        int indx = Indices[seid];
        float avg = AvgS[indx];
        float std = StdS[indx];

        // initialize diagonal pattern
        int p = W & 1;
        int q = (p == 0);
        
        // k % 2 and k / 2
        int km2 = 1, kd2 = 0;
        
        // row indices
        int trg_row = 1, pr1_row = 0, pr2_row = 2;
        
        // relax diagonal pattern
        for (int k = 2; k < 2*M+1; ++k) {

            // base index of row and associated shift
            kd2 += km2;
            km2 = (km2 == 0);
            int b_i = cntr+kd2+q*km2, b_j = -cntr+kd2+p*km2, shift = (p^km2);
            
            // cyclic indexing of rows
            pr1_row = trg_row;
            pr2_row = trg_row == 0 ? 2 : trg_row - 1;
            trg_row = trg_row == 2 ? 0 : trg_row + 1;
            
            for (int l = nidx; l < lane-1; l += get_local_size(0) ) {
                
                // potential index of node (if valid)
                int i = b_i-l;
                int j = b_j+l;
                
                // check if in valid relaxation area
                bool inside = 0 < i && i < M+1 && 0 < j && j < M+1 &&
                              shift <= l && l < lane-1;
                
                float value = INFINITY;
                
                if (inside) {
                    value = (Tsubject[indx+j-1]-avg)/std 
                          - sh_Query[i-1];
                    
                    // Euclidean or Manhattan distance?
                    value = euclidean ? value*value :
                            value < 0 ? -value : value;

                    // get mininum incoming edge and relax
                    value += min(sh_Penalty[pr2_row*lane+l],
                             min(sh_Penalty[pr1_row*lane+l-shift],
                                 sh_Penalty[pr1_row*lane+l-shift+1]));
                }
                
                // write value to penalty matrix
                sh_Penalty[trg_row*lane+l] = value;
            }
            // sync threads after each row
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // write result down
        Cdtw[seid] = sh_Penalty[((2*M) % 3)*lane+cntr];
    }
}


#define MAXQUERYSIZE (4096)
// linear memory constrained dtw (one dtw per thread)
__kernel void thread_cdtw(__global float * Subject, __global float * AvgS,__global float * StdS, __global float * Cdtw,
                 int M, __global int * Indices, int indices_l, int W,__global float * Tsubject, __global float * Czquery) {

    int thid = get_local_size(0) * get_group_id(0)  + get_local_id(0);
    
    if (thid < indices_l) {
        
        
        // retrieve actual index and lane length
        int indx = Indices[thid];
        int lane = M+1;
        
        // retrieve statistics of subject subsequence
        float avg = AvgS[indx];
        float std = StdS[indx];
        
        // linear memory algorithm M(n) = 2*(n+1) in O(n)
        float penalty[2*(MAXQUERYSIZE+1)];
       
        // initialize first row of penalty matrix
        penalty[0] = 0;
        for (int j = 1; j < lane; ++j)
            penalty[j] = INFINITY;

        // relax row-wise in topological sorted order
        for (int i = 1, src_row = 1, trg_row = 0; i < lane; ++i) {
            
            // calculate indices of source and target row
            trg_row = src_row;
            src_row = (trg_row == 0);
            int src_offset = src_row*lane;
            int trg_offset = trg_row*lane;
            float zquery_value = Czquery[i-1];
            
            // initialize target row
            for (int j = 0; j < lane; ++j)
                penalty[trg_offset+j] = INFINITY;
            
            // calculate Sakoe-Chiba band and relax nodes
            int lower = max(i-W, 1);
            int upper = min(i+W+1, lane);
           
            for (int j = lower; j < upper; ++j) {
                 
                 float value = zquery_value -
                                 (Tsubject[indx+j-1]-avg)/std;
                 
                 // Euclidean or Manhattan distance?
                 value = euclidean ? value*value :
                         value < 0 ? -value : value;
                 
                 // get nodes from the three incoming edges
                 value += min(penalty[src_offset+j-1], 
                          min(penalty[src_offset+j],
                              penalty[trg_offset+j-1]));
                 
                 // backrelax the three incoming edges
                 penalty[trg_offset+j] = value;
            }
        }
        
        // write down result
        Cdtw[thid] = penalty[(M & 1)*lane+M]; 
    }
}

#define FLT_MAX1 3.40282347e+38F

__kernel void reduction_min(__global float *input,__global float *per_block_results,int n, __local float * sdata)
{

	unsigned int i = get_local_size(0) * get_group_id(0)  + get_local_id(0);

	// load input into local memory
	float x = FLT_MAX1;
	if(i < n)
	{
		x = input[i];
	}
	sdata[get_local_id(0)] = x;
	barrier(CLK_LOCAL_MEM_FENCE);

    // contiguous range pattern
	for(int offset = get_local_size(0)  / 2;offset > 0; offset >>= 1)
	 {
	    if( get_local_id(0) < offset)
	    {
	        
	        float s = sdata[get_local_id(0)] ;
            float d = sdata[get_local_id(0) + offset];
            if ( s > d) sdata[get_local_id(0)] = d;
	    }

    // wait until all threads in the block have
    // updated their partial min
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // thread 0 writes the final result
    if(get_local_id(0) == 0)
    {
        per_block_results[get_group_id(0) ] = sdata[0];
    }
}




__kernel void random_lb_keogh(__global float * Subject, __global float * AvgS,__global float * StdS, 
                     __global float *  Lb_keogh_,int Lb_keogh_offset, int M, __global int *Indices_, int Indices_offset, 
                     int indices_l, __global float * Czlower, __global float * Czupper) {
    
    int thid = get_local_size(0) * get_group_id(0)  + get_local_id(0);
    __global float *  Lb_keogh =  &Lb_keogh_[Lb_keogh_offset];
    __global int *  Indices =  &Indices_[Indices_offset];
    // calculate LB_Keogh
    if (thid < indices_l) {
    
        int indx = Indices[thid];
    
        // obtain statistics
        float residues= 0;
        float avg = AvgS[indx];
        float std = StdS[indx];

        for (int i = 0; i < M; ++i) {
        
            // differences to envelopes
            float value = (Subject[indx+i]-avg)/std;
            float lower = value-Czlower[i];
            float upper = value-Czupper[i];
            
            // Euclidean or Manhattan distance?
            if (euclidean)
                residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
            else
                residues += upper*(upper > 0) - lower*(lower<0);
        }
        
        Lb_keogh[thid] = residues;
    }
}


__kernel void thresh (__global float * Series_, int offset, __global int * Mask, int L, float thr) {
    
    int thid = get_local_size(0) * get_group_id(0)  + get_local_id(0);
    __global float * Series = & Series_[offset];
    if (thid < L)
        Mask[thid] = (Series[thid] <= thr);
}


__kernel void merge (__global int * Mask, __global int * Prfx, int L) {
    
    int thid = get_local_size(0) * get_group_id(0)  + get_local_id(0);
    
    if (thid < L)
        Prfx[thid] *= Mask[thid];
}


__kernel void collect(__global int * Indices_, int Indices_offset, __global int * Mask, __global int * Prfx,int L) {

    int thid =  get_local_size(0) * get_group_id(0)  + get_local_id(0);
    __global int * Indices = &Indices_[Indices_offset];
    if (thid < L && Prfx[thid]) {
        Mask[Prfx[thid]-1] = Indices[thid];
    }

}


/////////////////////THESE KERNEL FOR RADIX SORT//////////////////////

#define _TOTALBITS 32  // number of bits for the integer in the list (max=32)
#define _BITS 4  // number of bits in the radix
#define _RADIX (1 << _BITS) //  radix  = 2^_BITS
#define PERMUT 1

// compute the histogram for each radix and each virtual processor for the pass
__kernel void histogram(const __global int* d_Keys,
			__global int* d_Histograms,
			const int pass,
			__local int* loc_histo,
			const int n){


  int it = get_local_id(0);
  int ig = get_global_id(0);

  int gr = get_group_id(0);

  int groups=get_num_groups(0);
  int items=get_local_size(0);

  // set the local histograms to zero
  for(int ir=0;ir<_RADIX;ir++){
    //d_Histograms[ir * groups * items + items * gr + it]=0;
    loc_histo[ir * items + it] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);


  // range of keys that are analyzed by the work item
  //int start= gr * n/groups + it * n/groups/items;
  int start= ig *(n/groups/items);
  int size= n/groups/items;

  int key,shortkey;

  for(int i= start; i< start + size;i++){
    key=d_Keys[i];

    // extract the group of _BITS bits of the pass
    // the result is in the range 0.._RADIX-1
    shortkey=(( key >> (pass * _BITS)) & (_RADIX-1));

    //d_Histograms[shortkey * groups * items + items * gr + it]++;
    loc_histo[shortkey *  items + it ]++;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // copy the local histogram to the global one
  for(int ir=0;ir<_RADIX;ir++){
    d_Histograms[ir * groups * items + items * gr + it]=loc_histo[ir * items + it];
  }

  barrier(CLK_GLOBAL_MEM_FENCE);


}

__kernel void transpose(const __global int* invect,
			__global int* outvect,
			const int nbcol,
			const int nbrow,
			const __global int* inperm,
			__global int* outperm,
			__local int* blockmat,
			__local int* blockperm,
			const int tilesize){

  int i0 = get_global_id(0)*tilesize;  // first row index
  int j = get_global_id(1);  // column index

  int iloc = 0;  // first local row index
  int jloc = get_local_id(1);  // local column index

  // fill the cache
  for(iloc=0;iloc<tilesize;iloc++){
    int k=(i0+iloc)*nbcol+j;  // position in the matrix
    blockmat[iloc*tilesize+jloc]=invect[k];
    if (PERMUT) {
      blockperm[iloc*tilesize+jloc]=inperm[k];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // first row index in the transpose
  int j0=get_group_id(1)*tilesize;

  // put the cache at the good place
  // loop on the rows
  for(iloc=0;iloc<tilesize;iloc++){
    int kt=(j0+iloc)*nbrow+i0+jloc;  // position in the transpose
    outvect[kt]=blockmat[jloc*tilesize+iloc];
    if (PERMUT) {
      outperm[kt]=blockperm[jloc*tilesize+iloc];
    }
  }

}

// each virtual processor reorders its data using the scanned histogram
__kernel void reorder(const __global int* d_inKeys,
		      __global int* d_outKeys,
		      __global int* d_Histograms,
		      const int pass,
		      __global int* d_inPermut,
		      __global int* d_outPermut,
		      __local int* loc_histo,
		      const int n){

  int it = get_local_id(0);
  int ig = get_global_id(0);

  int gr = get_group_id(0);
  int groups=get_num_groups(0);
  int items=get_local_size(0);

  int start= ig *(n/groups/items);
  int size= n/groups/items;

  // take the histograms in the cache
  for(int ir=0;ir<_RADIX;ir++){
    loc_histo[ir * items + it]=
      d_Histograms[ir * groups * items + items * gr + it];
  }
  barrier(CLK_LOCAL_MEM_FENCE);


  int newpos,ik,key,shortkey;

  for(int i= start; i< start + size;i++){
    key = d_inKeys[i];
    shortkey=((key >> (pass * _BITS)) & (_RADIX-1));
    newpos=loc_histo[shortkey * items + it];
    d_outKeys[newpos]= key;  // killing line !!!
    if(PERMUT) {
      d_outPermut[newpos]=d_inPermut[i];
    }
    newpos++;
    loc_histo[shortkey * items + it]=newpos;

  }


}


__kernel void scanhistograms( __global int* histo,__local int* temp,__global int* globsum){


  int it = get_local_id(0);
  int ig = get_global_id(0);
  int decale = 1;
  int n=get_local_size(0) * 2 ;
  int gr=get_group_id(0);

  // load input into local memory
  // up sweep phase
  temp[2*it] = histo[2*ig];
  temp[2*it+1] = histo[2*ig+1];

  // parallel prefix sum (algorithm of Blelloch 1990)
  for (int d = n>>1; d > 0; d >>= 1){
    barrier(CLK_LOCAL_MEM_FENCE);
    if (it < d){
      int ai = decale*(2*it+1)-1;
      int bi = decale*(2*it+2)-1;
      temp[bi] += temp[ai];
    }
    decale *= 2;
  }

  // store the last element in the global sum vector
  // (maybe used in the next step for constructing the global scan)
  // clear the last element
  if (it == 0) {
    globsum[gr]=temp[n-1];
    temp[n - 1] = 0;
  }

  // down sweep phase
  for (int d = 1; d < n; d *= 2){
    decale >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (it < d){
      int ai = decale*(2*it+1)-1;
      int bi = decale*(2*it+2)-1;

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }

  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // write results to device memory

  histo[2*ig] = temp[2*it];
  histo[2*ig+1] = temp[2*it+1];

  barrier(CLK_GLOBAL_MEM_FENCE);

}

// use the global sum for updating the local histograms
// each work item updates two values
__kernel void pastehistograms( __global int* histo,__global int* globsum){


  int ig = get_global_id(0);
  int gr=get_group_id(0);

  int s;

  s=globsum[gr];

  // write results to device memory
  histo[2*ig] += s;
  histo[2*ig+1] += s;

  barrier(CLK_GLOBAL_MEM_FENCE);

}

////////////////////END KERNEL PART OF RADIX SORT
