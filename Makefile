SHELL := /bin/bash

CC       = g++
FLAGS    = -O3 -g

CC_CU    = nvcc
FLAGS_CU = -O3 -g

CLANG    = /opt/opencilk/bin/clang++
FLAGS_CL = -O3 -fopencilk -fcilktool=cilkscale

seq: fglt_sequential.cpp
	$(CC) fglt_sequential.cpp $(FLAGS) -o fglt_seq.out

cuda: fglt_cuda.cu
	$(CC_CU) $(FLAGS_CU) fglt_cuda.cu -o fglt_cuda.out

tester: tester.cpp
	$(CC) tester.cpp $(FLAGS) -o tester.out