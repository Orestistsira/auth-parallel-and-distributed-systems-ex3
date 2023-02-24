#include "utils.hpp"

void compute_d4(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double *d4){

  double *fl = (double *) calloc(n, sizeof(double));
  int *pos = (int *) calloc(n, sizeof(int));
  int *isNgbh = (int *) calloc(n, sizeof(int));
  mwIndex *isUsed = (mwIndex *) calloc(n, sizeof(mwIndex));

  for(int i=0;i<n;i++){
    // setup the count of nonzero columns (j) visited for this row (i)
    mwIndex cnt = 0;

    // --- loop through every nonzero element A(i,k)
    for(mwIndex id_i = col[i]; id_i < col[i+1]; id_i++){

      // get the column (k)
      mwIndex k = row[id_i];

      isNgbh[k] = id_i+1;
      
      // --- loop through all nonzero elemnts A(k,j)
      for(mwIndex id_k = col[k]; id_k < col[k+1]; id_k++){

        // get the column (j)
        mwIndex j = row[id_k];

        if (i == j) continue;

        // if this column is not visited yet for this row (i), then set it
        if (!isUsed[j]) {
          fl[j]      = 0;  // initialize corresponding element
          isUsed[j]  = 1;  // set column as visited
          pos[cnt++] = j;  // add column position to list of visited
        }

        // increase count of A(i,j)
        fl[j]++;
          
      }

    }

    // --- perform reduction on [cnt] non-empty columns (j) 
    for(mwIndex l=0; l<cnt; l++) {

      // get next column number (j)
      mwIndex j = pos[l];

      if (isNgbh[j]) {
              
        d4[i]  += fl[j];
      }
      
      // declare it non-used
      isUsed[j] = 0;
    }

    d4[i]  /= 2;

    for(mwIndex id_i = col[i]; id_i < col[i+1]; id_i++){

      // get the column (k)
      mwIndex k = row[id_i];

      isNgbh[k] = 0;
    }

  }

  free(fl);
  free(pos);
  free(isNgbh);
  free(isUsed);
}

void compute_square(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double* d4){
  mwIndex ip = 0;

  //maybe 2 * m
  mwIndex *rowRes = (mwIndex*) malloc(m * sizeof(mwIndex));
  //mwIndex *colRes = (mwIndex*) calloc(n, sizeof(mwIndex));
  mwIndex colPoint = 0;

  int *xb = (int*) malloc(n * sizeof(int));
  for(mwIndex i = 0; i < n; i++){
    xb[i] = -1;
  }
  mwIndex *x = (mwIndex*) malloc(n * sizeof(mwIndex));

  //for each column (i)
  for(mwIndex i = 0; i < n; i++){
    colPoint = ip;

    //for each row (j) that column (i) has an element
    for(mwIndex jp = col[i]; jp < col[i+1];jp++){
      mwIndex j = row[jp];

      //for each row (k) that column (j) has an element
      for(mwIndex kp = col[j]; kp < col[j+1]; kp++){
        mwIndex k = row[kp];
        // printf("k = %d\n", k);

        //compute only if the value in starting matrix is 1
        bool compute = false;
        for(mwIndex fp = col[i]; fp < col[i+1]; fp++){
          mwIndex f = row[fp];
          if(f == k){
            compute = true;
            break;
          } 
        }

        if(!compute) continue;

        if(xb[k] != i){
          rowRes[ip] = k;
          ip += 1;
          xb[k] = i;
          //1 = valxval
          x[k] = 1;
        }
        else{
          //1 = valxval
          x[k] = x[k] + 1;
        }
      }
    }

    for(mwIndex vp = colPoint; vp < ip; vp++){
      mwIndex v = rowRes[vp];

      d4[i] += x[v];
    }

    d4[i] /= 2;
  }

  free(x);
  free(xb);
  free(rowRes);
}

void compute_c3(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double* d4){
  int j, jp, k, l, x;

  //for each column (i)
  for(int i = 0; i < n; i++){

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

void computeRaw(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double **d){
  //d0, d1
  for(mwSize i=0;i<n;i++){
    d[0][i] = 1;
    d[1][i] = col[i+1] - col[i];
    d[3][i] = d[1][i] * (d[1][i] - 1) * 0.5;
  }

  //d2, d3
  for(mwSize i=0;i<n;i++){
    for(mwIndex id_i = col[i]; id_i < col[i+1]; id_i++){

      // get the column (k)
      mwIndex k = row[id_i];
        
      // --- matrix-vector products
      d[2][i] += d[1][k];
    }

    d[2][i] -= d[1][i];
  }

  //d4
  compute_c3(row, col, n, m, d[4]);
}

int main(int argc, char **argv){

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
  //printArray(col, n + 1);

  //printArray(row, m);
  int graphletSize = 5;
  double **d  = (double **) malloc(graphletSize * sizeof(double *));

  for(int igraph = 0; igraph < graphletSize; igraph++){
    d[igraph] = (double *) calloc(n, sizeof(double));
  }

  struct timeval startwtime, endwtime;
  double duration;

  gettimeofday (&startwtime, NULL);
  computeRaw(row, col, n, m, d);
  gettimeofday (&endwtime, NULL);
  duration = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("[computeRaw took %.4f seconds]\n", duration);

  printf("[outputting results to \"raw_results.txt\"]\n");
  printResults(d, n, "raw_results.txt");

  for(int igraph = 0; igraph < graphletSize; igraph++){
    free(d[igraph]);
  }
  free(d);
  free(row);
  free(col);
  
}