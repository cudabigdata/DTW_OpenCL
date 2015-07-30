#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <unistd.h>

#include "include/cdtw.hpp"
#include "include/SystemCL.hpp"

#include "include/opencl_def.hpp"
#include <stdio.h>
#include <cstring>
//float * rs_cost = NULL;



typedef struct Path
{
  int k;
  int *px;
  int *py;

  Path(){
	  k = 0;
	  px = NULL;
	  py = NULL;
  }
} Path;

Path rs_path ;

//float rs_distance = INFINITY;

int path(float *cost, int n, int m, int startx, int starty, Path *p);
int subsequence_path(float *cost, int n, int m, Path *p, int offset);
float dtw_subsequence(float *x, float *y, int n, int m, float *cost);

#ifdef __cplusplus
extern "C" {
#endif



void get_path(int * rs_px, int * rs_py){
	for (int i = 0 ; i < rs_path.k; i++)
	{
		rs_px[i] = rs_path.px[i];
		rs_py[i] = rs_path.py[i];
	}
}
int get_path_length(){
	return rs_path.k;
}

float gpu_dtw_func(float * zquery, float * subject, int M, int N,
		float * rs_cost) {

	printf("Calling GPU-DTW Function...\n");
	float *zlower = NULL, *zupper = NULL;

	cl_mem Subject = NULL;
	cl_mem AvgS = NULL;
	cl_mem StdS = NULL;

	int W = M * 0.4;

	// host side memory
	zlower = (float*) malloc(sizeof(float) * M);
	zupper = (float*) malloc(sizeof(float) * M);

	cl_context context = SystemCL::Inst().Context();
	cl_int ret;

	// device side memory

	Subject = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N, NULL,
			&ret);
	CLERR(ret);
	AvgS = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(float) * (N - M + 1), NULL, &ret);
	CLERR(ret);
	StdS = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(float) * (N - M + 1), NULL, &ret);
	CLERR(ret);

	// z-normalize query and envelope
	znormalize(zquery, M);
	envelope(zquery, W, zlower, zupper, M);

	// copy subject to gpu
	cl_command_queue queue = SystemCL::Inst().Queue();

	ret = clEnqueueWriteBuffer(queue, Subject, CL_TRUE, 0, sizeof(float) * N,
			subject, 0, NULL, NULL);
	CLERR(ret)

	// copy query and associated envelopes to constant memory
	Czlower = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M, NULL,
			&ret);
	CLERR(ret);
	Czupper = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M, NULL,
			&ret);
	CLERR(ret);
	Czquery = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M, NULL,
			&ret);
	CLERR(ret);

	ret = clEnqueueWriteBuffer(queue, Czlower, CL_TRUE, 0, sizeof(float) * M,
			zlower, 0, NULL, NULL);
	CLERR(ret)
	ret = clEnqueueWriteBuffer(queue, Czupper, CL_TRUE, 0, sizeof(float) * M,
			zupper, 0, NULL, NULL);
	CLERR(ret)
	ret = clEnqueueWriteBuffer(queue, Czquery, CL_TRUE, 0, sizeof(float) * M,
			zquery, 0, NULL, NULL);
	CLERR(ret)

	// calculate windowed average and standard deviation of Subject
	avg_std(Subject, AvgS, StdS, M, N);


	int matched_location = prune_cdtw(Subject, AvgS, StdS, M, N, W);

	printf("Matched location %d\n", matched_location);



	// Get need data
	znormalize(subject + matched_location, M);

	float rs_distance = dtw_subsequence(zquery, subject + matched_location, M, M,
			rs_cost);

	subsequence_path(rs_cost, M, M, &rs_path, matched_location);


	return rs_distance;
}

#ifdef __cplusplus
}
#endif

int main(int argc, char* argv[]) {


     if (argc != 6){
        std::cout << "call" << argv[0] 
                  << " query.bin subject.bin M N P" << std::endl; 
        return 1;
    }

    float *zlower = NULL, *zupper = NULL, *zquery = NULL, *subject = NULL;

    cl_mem Subject = NULL;
    cl_mem AvgS = NULL;
    cl_mem StdS = NULL;

    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    int W = M*(atoi(argv[5])*0.01);
    
    std::cout << "\n= info =====================================" << std::endl;
    std::cout << "|Query| = " << M << "\t"
              << "|Subject| = " << N << "\t"
              << "window = " << W << std::endl;

    // host side memory
    zlower = (float*) malloc(sizeof(float)*M);
    zupper = (float*) malloc(sizeof(float)*M);
    zquery = (float*) malloc(sizeof(float)*M);
    subject = (float*) malloc(sizeof(float)*N);

    cl_context context = SystemCL::Inst().Context();
    cl_int ret;

    // device side memory

    Subject = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)* N, NULL, &ret);        CLERR(ret);
    AvgS = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(N-M+1), NULL, &ret);    CLERR(ret);
    StdS = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(N-M+1), NULL, &ret);    CLERR(ret);

    struct timeval start, end;
    float secs_used,micros_used;

    std::cout << "\n= loading data =============================" << std::endl;
    
    gettimeofday(&start, NULL);
    
    // read query from file
    std::ifstream qfile(argv[1], std::ios::binary|std::ios::in);
    qfile.read((char *) &zquery[0], sizeof(float)*M);

    // read subject from file
    std::ifstream sfile(argv[2], std::ios::binary|std::ios::in);
    sfile.read((char *) &subject[0], sizeof(float)*N);

    // z-normalize query and envelope
    znormalize(zquery, M);
    envelope(zquery, W, zlower, zupper, M);

    gettimeofday(&end, NULL);;
    secs_used =(end.tv_sec - start.tv_sec);
    micros_used= (secs_used*1000) + (end.tv_usec - start.tv_usec) * 0.001;

    std::cout << "Miliseconds to load data: " << micros_used << std::endl;
    

    gettimeofday(&start, NULL);
    // copy subject to gpu
    cl_command_queue queue = SystemCL::Inst().Queue();

    ret = clEnqueueWriteBuffer(queue, Subject, CL_TRUE, 0,
    		 sizeof(float)*N, subject, 0, NULL, NULL);   CLERR(ret)


    // copy query and associated envelopes to constant memory
    Czlower = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*M, NULL, &ret);  CLERR(ret);
    Czupper = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*M, NULL, &ret);  CLERR(ret);
    Czquery = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*M, NULL, &ret);  CLERR(ret);

    ret = clEnqueueWriteBuffer(queue, Czlower, CL_TRUE, 0,
    		 sizeof(float)*M, zlower, 0, NULL, NULL);   CLERR(ret)
    ret = clEnqueueWriteBuffer(queue, Czupper, CL_TRUE, 0,
    		 sizeof(float)*M, zupper, 0, NULL, NULL);   CLERR(ret)
    ret = clEnqueueWriteBuffer(queue, Czquery, CL_TRUE, 0,
    		 sizeof(float)*M, zquery, 0, NULL, NULL);   CLERR(ret)

    // calculate windowed average and standard deviation of Subject
    avg_std(Subject, AvgS, StdS, M, N);

    std::cout << "\n= pruning ==================================" << std::endl;
    
    int matched_location = prune_cdtw(Subject, AvgS, StdS, M, N, W);

    std::cout << "Matched location " << matched_location<< std::endl;


    gettimeofday(&end, NULL);;
    secs_used =(end.tv_sec - start.tv_sec);
    micros_used= (secs_used*1000) + (end.tv_usec - start.tv_usec) * 0.001;

    std::cout << "Miliseconds to find best match: " << micros_used << std::endl;

    return 0;
}




float dist(float x, float y) {
	return pow(x - y, 2);
}

float min3(float a, float b, float c) {
	float min;

	min = a;
	if (b < min)
		min = b;
	if (c < min)
		min = c;
	return min;
}

float dtw_subsequence(float *x, float *y, int n, int m, float *cost) {
	 int i, j;

	  cost[0] = dist(x[0],y[0]);

	  for (i=1; i<n; i++)
	    cost[i*m] = dist(x[i],y[0]) + cost[(i-1)*m];

	  for (j=1; j<m; j++)
	    cost[j] = dist(x[0],y[j]); // subsequence variation: D(0,j) := c(x0, yj)

	  for (i=1; i<n; i++)
	    for (j=1; j<m; j++)
	      cost[i*m+j] = dist(x[i],y[j]) +
		min3(cost[(i-1)*m+j], cost[(i-1)*m+(j-1)], cost[i*m+(j-1)]);

	return cost[n * m - 1];
}

int path(float *cost, int n, int m, int startx, int starty, Path *p)
{
  int i, j, k, z1, z2;
  int *px;
  int *py;
  float min_cost;

  if ((startx >= n) || (starty >= m))
    return 0;

  if (startx < 0)
    startx = n - 1;

  if (starty < 0)
    starty = m - 1;

  i = startx;
  j = starty;
  k = 1;
  // allocate path for the worst case
  px = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));
  py = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));

  px[0] = i;
  py[0] = j;
  while ((i > 0) || (j > 0))
    {
      if (i == 0)
	j--;
      else if (j == 0)
	i--;
      else
	{
	  min_cost = min3(cost[(i-1)*m+j],
			  cost[(i-1)*m+(j-1)],
			  cost[i*m+(j-1)]);

	  if (cost[(i-1)*m+(j-1)] == min_cost)
	    {
	      i--;
	      j--;
	    }
	  else if (cost[i*m+(j-1)] == min_cost)
	    j--;
	  else
	    i--;
	}

      px[k] = i;
      py[k] = j;
      k++;
    }
  p->px = (int *) malloc (k * sizeof(int));
  p->py = (int *) malloc (k * sizeof(int));
  for (z1=0, z2=k-1; z1<k; z1++, z2--)
    {
      p->px[z1] = px[z2];
      p->py[z1] = py[z2];
    }
  p->k = k;
  free(px);
  free(py);

  return 1;
}


int
subsequence_path(float *cost, int n, int m, Path *p, int offset)
{
  int i, z1, z2;
  int a_star;
  int *tmpx, *tmpy;
  // find path
  if (!path(cost, n, m, -1, -1, p))
    return 0;
  // find a_star
  a_star = 0;
  for (i=1; i<p->k; i++)
    if (p->px[i] == 0)
      a_star++;
    else
      break;

  // rebuild path
  tmpx = p->px;
  tmpy = p->py;
  p->px = (int *) malloc ((p->k-a_star) * sizeof(int));
  p->py = (int *) malloc ((p->k-a_star) * sizeof(int));
  for (z1=0, z2=a_star; z2<p->k; z1++, z2++)
    {
      p->px[z1] = tmpx[z2];
      p->py[z1] = tmpy[z2] + offset;
    }
  p->k = p->k-a_star;

  free(tmpx);
  free(tmpy);
  return 1;
}
