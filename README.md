# auth-parallel-and-distributed-systems-ex3

**Accelerating FGLT library with CUDA**

---

CUDA Version: 12.0

*Compile and Run the code*

- To run sequential code:

```

make seq

./fglt_seq.out <filepath>

```

- To CUDA parallel code:

```

make cuda

./fglt_cuda.out <filepath>

```

- To compare the two result files:

```

make tester

./tester.out

```
