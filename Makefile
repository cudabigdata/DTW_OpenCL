CC= g++

OPT= -O2
 
LIB= -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lOpenCL

all: GPU-DTW GPU-DTW-LIB

GPU-DTW: GPU-dtw.cpp
	$(CC) $(OPT)  GPU-dtw.cpp -o GPU-DTW $(LIB)


GPU-DTW-LIB: GPU-dtw.cpp
	$(CC) -shared -fPIC  GPU-dtw.cpp -o GPU-DTW.so $(LIB)



clean:
	rm -rf GPU-DTW

