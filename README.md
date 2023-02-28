# auth-parallel-and-distributed-systems-ex3

**Accelerating FGLT library with CUDA**

---

CUDA Version: 12.0

*Compile and Run the code*

- To run sequential code:

```

make seq

./fglt_seq.out <grpah-filepath>

```

- To CUDA parallel code:

```

make cuda

./fglt_cuda.out <graph-filepath>

```

- To compare the two result files (Make sure to run the same graph file for the above):

```

make tester

./tester.out

```
