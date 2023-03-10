#include "utils.hpp"

#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess){
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define BLOCK_SIZE 512

__global__ void compute_d0_d1_d3_Kernel(mwIndex *col, double *d0, double *d1, double *d3, mwSize n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n){
    d0[i] = 1;
    d1[i] = col[i+1] - col[i];
    d3[i] = d1[i] * (d1[i] - 1) * 0.5;
  }
}

//matrix-vector multiplication
__global__ void compute_d2_Kernel(mwIndex *row, mwIndex *col, double *d1, double *d2, mwSize n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n){
    double sum = 0;
    for(mwIndex id_i = col[i]; id_i < col[i+1]; id_i++){

      // get the column (k)
      mwIndex k = row[id_i];
        
      // --- matrix-vector products
      sum += d1[k];
    }
    d2[i] = sum - d1[i];
  }
}

__device__ double warp_reduce(double sum){
  unsigned FULL_WARP_MASK = 0xffffffff;

  for(int offset = warpSize / 2; offset > 0; offset /= 2){
    sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
  }
  return sum;
}

__global__ void compute_d2_vector_Kernel(mwIndex *row, mwIndex *col, double *d1, double *d2, mwSize n){
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = threadId / 32;
  int lane = threadId % 32;

  //the column (one warp per column)
  mwIndex j = warpId;

  double sum = 0;
  if(j < n){
    
    for(mwIndex id_i = col[j] + lane; id_i < col[j+1]; id_i += 32){

      // get the column (k)
      mwIndex k = row[id_i];
        
      // --- matrix-vector products
      sum += d1[k];
    }
  }

  sum = warp_reduce(sum);

  if(lane == 0 && j < n)
    d2[j] = sum - d1[j];
}

__global__ void compute_c3_Kernel(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double* d4){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int j, jp, k, l, x;

  //for each column (i)
  if(i < n){
    //for each row (j) that column (i) has an element
    for(jp = col[i]; jp < col[i+1]; jp++){
      j = row[jp];
      x = col[i];

      for(k = col[j]; k < col[j + 1]; k++){
        for(l = x; l < col[i+1]; l++){

          if(row[k] < row[l]){
            x = l;
            break;
          }
          else if(row[k] == row[l]){
            d4[i]++;
            x = l + 1;
            break;
          }
          else{
            x = l + 1;
          }
        }
      }
    }

    d4[i] /= 2;
  }
}


void cudaComputeRaw(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double **d){
  double *d0, *d1, *d2, *d3, *d4;

  d0 = d[0];
  d1 = d[1];
  d2 = d[2];
  d3 = d[3];
  d4 = d[4];

  double *d0D, *d1D, *d2D, *d3D, *d4D;

  //d0, d1, d3------------------------------------

  CHECK_CUDA( cudaMalloc((void**)&d0D, n * sizeof(double)) );
  CHECK_CUDA( cudaMalloc((void**)&d1D, n * sizeof(double)) );
  CHECK_CUDA( cudaMalloc((void**)&d3D, n * sizeof(double)) );

  mwIndex *colD;
  CHECK_CUDA( cudaMalloc((void**)&colD, (n+1) * sizeof(mwIndex)) );
  CHECK_CUDA( cudaMemcpy(colD, col, (n+1) * sizeof(mwIndex), cudaMemcpyHostToDevice) );

  dim3 dimBlock(BLOCK_SIZE); //thread per block
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE); //num of blocks
  compute_d0_d1_d3_Kernel<<<dimGrid, dimBlock>>>(colD, d0D, d1D, d3D, n);

  CHECK_CUDA( cudaMemcpy(d0, d0D, n * sizeof(double), cudaMemcpyDeviceToHost) );
  CHECK_CUDA( cudaMemcpy(d1, d1D, n * sizeof(double), cudaMemcpyDeviceToHost) );
  CHECK_CUDA( cudaMemcpy(d3, d3D, n * sizeof(double), cudaMemcpyDeviceToHost) );

  CHECK_CUDA( cudaFree(d0D) );
  CHECK_CUDA( cudaFree(d3D) );

  //d2------------------------------------

  CHECK_CUDA( cudaMalloc((void**)&d2D, n * sizeof(double)) );
  
  mwIndex *rowD;
  CHECK_CUDA( cudaMalloc((void**)&rowD, m * sizeof(double)) );

  CHECK_CUDA( cudaMemcpy(rowD, row, m * sizeof(double), cudaMemcpyHostToDevice) );

  //------------if d2 vector kernel
  // dim3 dimBlock2(BLOCK_SIZE); //thread per block
  // dim3 dimGrid2(n); //num of blocks (and warps)
  //----------------------
  compute_d2_Kernel<<<dimGrid, dimBlock>>>(rowD, colD, d1D, d2D, n);
  //compute_d2_vector_Kernel<<<dimGrid2, dimBlock2>>>(rowD, colD, d1D, d2D, n);

  CHECK_CUDA( cudaMemcpy(d2, d2D, n * sizeof(double), cudaMemcpyDeviceToHost) );

  //d4
  CHECK_CUDA( cudaMalloc((void**)&d4D, n * sizeof(double)) );

  compute_c3_Kernel<<<dimGrid, dimBlock>>>(rowD, colD, n, m, d4D);

  CHECK_CUDA( cudaMemcpy(d4, d4D, n * sizeof(double), cudaMemcpyDeviceToHost) );

  CHECK_CUDA( cudaFree(d1D) );
  CHECK_CUDA( cudaFree(d2D) );
  CHECK_CUDA( cudaFree(d4D) );
  CHECK_CUDA( cudaFree(colD) );
  CHECK_CUDA( cudaFree(rowD) );
}

int main(int argc, char **argv)
{

  // ~~~~~~~~~~ variable declarations
  mwIndex *row, *col;
  mwSize  n, m;
  std::string filename = "graph.mtx";

  // ~~~~~~~~~~ parse inputs

  // ----- retrieve the (non-option) argument:
  if ( (argc <= 1) || (argv[argc-1] == NULL) || (argv[argc-1][0] == '-') ) {
    // there is NO input...
    std::cout << "No filename provided, using 'graph.mtx'." << std::endl;
  }
  else {
    // there is an input...
    filename = argv[argc-1];
    std::cout << "Using graph stored in '" << filename << "'." << std::endl;
  }
  
  readMTX(&row, &col, &n, &m, filename.c_str());

  printf("number of columns = %ld\n", n);
  printf("non zero elements = %ld\n", m);

  int graphletSize = 5;
  double **d  = (double **) malloc(graphletSize * sizeof(double *));

  for(int igraph = 0; igraph < graphletSize; igraph++){
    d[igraph] = (double *) calloc(n, sizeof(double));
  }

  struct timeval startwtime, endwtime;
  double duration;

  gettimeofday (&startwtime, NULL);
  cudaComputeRaw(row, col, n, m, d);
  gettimeofday (&endwtime, NULL);
  duration = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("[cudaComputeRaw took %.4f seconds]\n", duration);

  printf("[outputting results to \"raw_results_cuda.txt\"]\n");
  printResults(d, n, "raw_results_cuda.txt");

  for(int igraph = 0; igraph < graphletSize; igraph++){
    free(d[igraph]);
  }
  free(d);
  free(row);
  free(col);
  
}