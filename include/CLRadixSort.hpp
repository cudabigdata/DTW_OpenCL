// C++ class for sorting integer list in OpenCL
// copyright Philippe Helluy, Université de Strasbourg, France, 2011, helluy@math.unistra.fr
// licensed under the GNU Lesser General Public License see http://www.gnu.org/copyleft/lesser.html
// if you find this software usefull you can cite the following work in your reports or articles:
// Philippe HELLUY, A portable implementation of the radix sort algorithm in OpenCL, HAL 2011.
// The algorithm is the radix sort algorithm
// Marcho Zagha and Guy E. Blelloch. “Radix Sort For Vector Multiprocessor.”
// in: Conference on High Performance Networking and Computing, pp. 712-721, 1991.
// each integer is made of _TOTALBITS bits. The radix is made of _BITS bits. The sort is made of
// several passes, each consisting in sorting against a group of bits corresponding to the radix.
// _TOTALBITS/_BITS passes are needed.
// The sorting parameters can be changed in "CLRadixSortParam.hpp"
// compilation for Mac:
//g++ CLRadixSort.cpp CLRadixSortMain.cpp -framework opencl -Wall
// compilation for Linux:
//g++ CLRadixSort.cpp CLRadixSortMain.cpp -lOpenCL -Wall

#ifndef _CLRADIXSORT
#define _CLRADIXSORT

#include "SystemCL.hpp"

#include <string>
#include<fstream>
#include<iostream>
#include<math.h>
#include <stdlib.h>

using namespace std;

#define _ITEMS  16 // number of items in a group
#define _GROUPS 16 // the number of virtual processors is _ITEMS * _GROUPS
#define  _HISTOSPLIT 512 // number of splits of the histogram
#define _TOTALBITS 32  // number of bits for the integer in the list (max=32)
#define _BITS 4  // number of bits in the radix
#define PERMUT 1  // store the final permutation
// the following parameters are computed from the previous
#define _RADIX (1 << _BITS) //  radix  = 2^_BITS
#define _PASS (_TOTALBITS/_BITS) // number of needed passes to sort the list
#define _HISTOSIZE (_ITEMS * _GROUPS * _RADIX ) // size of the histogram

#define _MAXINT (1 << (_TOTALBITS-1))

class CLRadixSort{



public:
  CLRadixSort(cl_mem key, cl_mem index, int L){

		Context = SystemCL::Inst().Context();
		CommandQueue = SystemCL::Inst().Queue();

		ckHistogram = SystemCL::Inst().ckHistogram;
		ckScanHistogram = SystemCL::Inst().ckScanHistogram;
		ckPasteHistogram = SystemCL::Inst().ckPasteHistogram;
		ckReorder = SystemCL::Inst().ckReorder;
		ckTranspose = SystemCL::Inst().ckTranspose;

	  nkeys = L;
	  nkeys_rounded= ((nkeys - 1)/(_ITEMS * _GROUPS) + 1) * (_ITEMS * _GROUPS);

	  cl_int err;
	  d_inKeys  = clCreateBuffer(Context,
				     CL_MEM_READ_WRITE,
				     sizeof(uint)* nkeys_rounded ,
				     NULL,
				     &err); CLERR(err)

	  d_outKeys  = clCreateBuffer(Context,
				      CL_MEM_READ_WRITE,
				      sizeof(uint)* nkeys_rounded ,
				      NULL,
				      &err);  CLERR(err)

	  d_inPermut  = clCreateBuffer(Context,
				       CL_MEM_READ_WRITE,
				       sizeof(uint)* nkeys_rounded ,
				       NULL,
				       &err);  CLERR(err)

	  d_outPermut  = clCreateBuffer(Context,
					CL_MEM_READ_WRITE,
					sizeof(uint)* nkeys_rounded ,
					NULL,
					&err);  CLERR(err)

	 err = clEnqueueCopyBuffer(CommandQueue, key, d_inKeys, 0, 0, sizeof(uint)  * L, 0, NULL, NULL);CLERR(err)
	 err = clEnqueueCopyBuffer(CommandQueue, index, d_inPermut, 0, 0, sizeof(uint)  * L, 0, NULL, NULL);CLERR(err)


	  // allocate the histogram on the GPU
	  d_Histograms  = clCreateBuffer(Context,
					 CL_MEM_READ_WRITE,
					 sizeof(uint)* _RADIX * _GROUPS * _ITEMS,
					 NULL,
					 &err);  CLERR(err)



	  // allocate the auxiliary histogram on GPU
	  d_globsum  = clCreateBuffer(Context,
				      CL_MEM_READ_WRITE,
				      sizeof(uint)* _HISTOSPLIT,
				      NULL,
				      &err);  CLERR(err)

	  // temporary vector when the sum is not needed
	  d_temp  = clCreateBuffer(Context,
				   CL_MEM_READ_WRITE,
				   sizeof(uint)* _HISTOSPLIT,
				   NULL,
				   &err);  CLERR(err)

	  Resize(nkeys);


	  // we set here the fixed arguments of the OpenCL kernels
	  // the changing arguments are modified elsewhere in the class

	  err = clSetKernelArg(ckHistogram, 1, sizeof(cl_mem), &d_Histograms);  CLERR(err)


	  err = clSetKernelArg(ckHistogram, 3, sizeof(uint)*_RADIX*_ITEMS, NULL);  CLERR(err)

	  err = clSetKernelArg(ckPasteHistogram, 0, sizeof(cl_mem), &d_Histograms);  CLERR(err)


	  err = clSetKernelArg(ckPasteHistogram, 1, sizeof(cl_mem), &d_globsum);  CLERR(err)

	  err = clSetKernelArg(ckReorder, 2, sizeof(cl_mem), &d_Histograms);  CLERR(err)

	  err  = clSetKernelArg(ckReorder, 6,
				sizeof(uint)* _RADIX * _ITEMS ,
				NULL); CLERR(err)

	}
  
  CLRadixSort() {};
  ~CLRadixSort(){
  clReleaseMemObject(d_inKeys);
  clReleaseMemObject(d_outKeys);
  clReleaseMemObject(d_Histograms);
  clReleaseMemObject(d_globsum);
  clReleaseMemObject(d_inPermut);
  clReleaseMemObject(d_outPermut);
  }
  

  // this function allows to change the size of the sorted vector
  void Resize(int nn){

	  // length of the vector has to be divisible by (_GROUPS * _ITEMS)
	  int reste=nkeys % (_GROUPS * _ITEMS);
	  nkeys_rounded=nkeys;
	  cl_int err;
	  unsigned int pad[_GROUPS * _ITEMS];
	  for(int ii=0;ii<_GROUPS * _ITEMS;ii++){
	    pad[ii]= _MAXINT ;
	  }
	  if (reste !=0) {
	    nkeys_rounded=nkeys-reste+(_GROUPS * _ITEMS);
	    // pad the vector with big values
	    err = clEnqueueWriteBuffer(CommandQueue,
				       d_inKeys,
				       CL_TRUE, sizeof(uint)*nkeys,
				       sizeof(uint) *(_GROUPS * _ITEMS - reste) ,
				       pad,
				       0, NULL, NULL);

	  }

	}

  // this function treats the array d_Keys on the GPU
  // and return the sorting permutation in the array d_Permut
  void Sort(){


	  int nbcol=nkeys_rounded/(_GROUPS * _ITEMS);
	  int nbrow= _GROUPS * _ITEMS;

	  for(uint pass=0;pass<_PASS;pass++){


	    Histogram(pass);

	    ScanHistogram();

	    Reorder(pass);
	  }


	}

  // get the data from the GPU (for debugging)
  void RecupGPU(cl_mem key, cl_mem index, int L ){
	     cl_int err;
		 err = clEnqueueCopyBuffer(CommandQueue, d_inKeys, key, 0, 0, sizeof(uint)  * L, 0, NULL, NULL);CLERR(err)
		 err = clEnqueueCopyBuffer(CommandQueue, d_inPermut,index, 0, 0, sizeof(uint)  * L, 0, NULL, NULL);CLERR(err)

	}
  // compute the histograms for one pass
  void Histogram(uint pass){

	  cl_int err;

	  size_t nblocitems=_ITEMS;
	  size_t nbitems=_GROUPS*_ITEMS;


	  err  = clSetKernelArg(ckHistogram, 0, sizeof(cl_mem), &d_inKeys); CLERR(err)

	  err = clSetKernelArg(ckHistogram, 2, sizeof(uint), &pass); CLERR(err)

	  err = clSetKernelArg(ckHistogram, 4, sizeof(uint), &nkeys_rounded); CLERR(err)

	  err = clEnqueueNDRangeKernel(CommandQueue,
				       ckHistogram,
				       1, NULL,
				       &nbitems,
				       &nblocitems,
				       0, NULL, NULL); CLERR(err)

	  clFinish(CommandQueue);


	}
  // scan the histograms
  void ScanHistogram(void){

	  cl_int err;

	  // numbers of processors for the local scan
	  // half the size of the local histograms
	  size_t nbitems=_RADIX* _GROUPS*_ITEMS / 2;


	  size_t nblocitems= nbitems/_HISTOSPLIT ;


	  int maxmemcache=max(_HISTOSPLIT,_ITEMS * _GROUPS * _RADIX / _HISTOSPLIT);

	  // scan locally the histogram (the histogram is split into several
	  // parts that fit into the local memory)

	  err = clSetKernelArg(ckScanHistogram, 0, sizeof(cl_mem), &d_Histograms);  CLERR(err)

	  err  = clSetKernelArg(ckScanHistogram, 1,
				sizeof(uint)* maxmemcache ,
				NULL);  CLERR(err)

	  err = clSetKernelArg(ckScanHistogram, 2, sizeof(cl_mem), &d_globsum); CLERR(err)


	  err = clEnqueueNDRangeKernel(CommandQueue,
				       ckScanHistogram,
				       1, NULL,
				       &nbitems,
				       &nblocitems,
				       0, NULL, NULL);

	  clFinish(CommandQueue);


	  // second scan for the globsum
	  err = clSetKernelArg(ckScanHistogram, 0, sizeof(cl_mem), &d_globsum); CLERR(err)
	  err = clSetKernelArg(ckScanHistogram, 2, sizeof(cl_mem), &d_temp); CLERR(err)

	  nbitems= _HISTOSPLIT / 2;
	  nblocitems=nbitems;

	  err = clEnqueueNDRangeKernel(CommandQueue,
	  			       ckScanHistogram,
	  			       1, NULL,
	  			       &nbitems,
	  			       &nblocitems,
	  			       0, NULL, NULL); CLERR(err)

	  clFinish(CommandQueue);

	  // loops again in order to paste together the local histograms
	  nbitems = _RADIX* _GROUPS*_ITEMS/2;
	  nblocitems=nbitems/_HISTOSPLIT;

	  err = clEnqueueNDRangeKernel(CommandQueue,
	  			       ckPasteHistogram,
	  			       1, NULL,
	  			       &nbitems,
	  			       &nblocitems,
	  			       0, NULL, NULL); CLERR(err)

	  clFinish(CommandQueue);



	}
  // scan the histograms
  void Reorder(uint pass){


	  cl_int err;

	  size_t nblocitems=_ITEMS;
	  size_t nbitems=_GROUPS*_ITEMS;


	  clFinish(CommandQueue);

	  err  = clSetKernelArg(ckReorder, 0, sizeof(cl_mem), &d_inKeys); CLERR(err)

	  err  = clSetKernelArg(ckReorder, 1, sizeof(cl_mem), &d_outKeys); CLERR(err)

	  err = clSetKernelArg(ckReorder, 3, sizeof(uint), &pass); CLERR(err)

	  err  = clSetKernelArg(ckReorder, 4, sizeof(cl_mem), &d_inPermut); CLERR(err)

	  err  = clSetKernelArg(ckReorder, 5, sizeof(cl_mem), &d_outPermut); CLERR(err)

	  err  = clSetKernelArg(ckReorder, 6,
				sizeof(uint)* _RADIX * _ITEMS ,
				NULL);  CLERR(err)


	  err = clSetKernelArg(ckReorder, 7, sizeof(uint), &nkeys_rounded); CLERR(err)


	  err = clEnqueueNDRangeKernel(CommandQueue,
				       ckReorder,
				       1, NULL,
				       &nbitems,
				       &nblocitems,
				       0, NULL, NULL); CLERR(err)

	  clFinish(CommandQueue);

	  // swap the old and new vectors of keys
	  cl_mem d_temp;
	  d_temp=d_inKeys;
	  d_inKeys=d_outKeys;
	  d_outKeys=d_temp;

	  // swap the old and new permutations
	  d_temp=d_inPermut;
	  d_inPermut=d_outPermut;
	  d_outPermut=d_temp;

	}


  cl_context Context;             // OpenCL context

  cl_command_queue CommandQueue;     // OpenCL command queue 

  uint h_Histograms[_RADIX * _GROUPS * _ITEMS]; // histograms on the cpu
  cl_mem d_Histograms;                   // histograms on the GPU

  // sum of the local histograms
  uint h_globsum[_HISTOSPLIT];
  cl_mem d_globsum;
  cl_mem d_temp;  // in case where the sum is not needed

  // list of keys
  uint nkeys; // actual number of keys
  uint nkeys_rounded; // next multiple of _ITEMS*_GROUPS

  cl_mem d_inKeys;
  cl_mem d_outKeys;

  // permutation
  cl_mem d_inPermut;
  cl_mem d_outPermut;

   // OpenCL kernels
  cl_kernel ckTranspose; // transpose the initial list
  cl_kernel ckHistogram;  // compute histograms
  cl_kernel ckScanHistogram; // scan local histogram
  cl_kernel ckPasteHistogram; // paste local histograms
  cl_kernel ckReorder; // final reordering

};

#endif
