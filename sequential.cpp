#include <iostream>
#include <fstream>
#include <cassert>
#include <limits>
#include <string.h>

#include "fglt.hpp"

void readMTX
(
 mwIndex       **       row,
 mwIndex       **       col,
 mwSize         * const n,
 mwSize         * const m,
 char    const  * const filename
){

  // ~~~~~~~~~~ variable declarations
  mwIndex *row_coo, *col_coo;
  mwSize  n_col, m_mat;
  char mmx[20], b1[20], b2[20], b3[20], b4[20];
  bool issymmetric = false;
  
  // ~~~~~~~~~~ read matrix
  
  // open the file
  std::ifstream fin( filename );

  // check if file exists
  if( fin.fail() ){
    std::cerr << "File " << filename << " could not be opened! Aborting..." << std::endl;
    exit(1);
  }

  // read banner
  fin >> mmx >> b1 >> b2 >> b3 >> b4;

  // parse banner
  if ( strcmp( b1, "matrix" ) ){
    std::cerr << "Currently works only with 'matrix' option, aborting..." << std::endl;
    exit(1);
  }

  if ( strcmp( b2, "coordinate" ) ){
    std::cerr << "Currently works only with 'coordinate' option, aborting..." << std::endl;
    exit(1);
  }
  
  if ( strcmp( b3, "pattern" ) ){
    std::cerr << "Currently works only with 'pattern' format, aborting..." << std::endl;
    exit(1);
  }

  if ( !strcmp( b4, "symmetric" ) ){
    issymmetric = true;
  }



  // ignore headers and comments
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  // read defining parameters
  fin >> n[0] >> n_col >> m[0];

  if (issymmetric) m_mat = 2*m[0];
  else             m_mat = m[0];
  
  assert( n[0] == n_col );

  // allocate space for COO format
  row_coo = static_cast<mwIndex *>( malloc(m_mat * sizeof(mwIndex)) );
  col_coo = static_cast<mwIndex *>( malloc(m_mat * sizeof(mwIndex)) );
  
  // read the COO data
  mwIndex k = 0;
  for (mwIndex l = 0; l < m[0]; l++){
    fin >> row_coo[k] >> col_coo[k];
    if (issymmetric)
      if (row_coo[k] == col_coo[k])  m_mat -= 2; // we do not keep self-loop, remove edges
      else { row_coo[k+1] = col_coo[k]; col_coo[k+1] = row_coo[k]; k += 2; } // put symmetric edge
    else
      if (row_coo[k] == col_coo[k])  m_mat -= 1; // we do not keep self-loop, remove edge
      else k++;
  }

  m[0] = m_mat;
  row_coo =static_cast<mwIndex *>( realloc( row_coo, m[0] * sizeof(mwIndex) ) );
  col_coo =static_cast<mwIndex *>( realloc( col_coo, m[0] * sizeof(mwIndex) ) );

  // close connection to file
  fin.close();

  // ~~~~~~~~~~ transform COO to CSC
  row[0] = static_cast<mwIndex *>( malloc( m[0]   * sizeof(mwIndex)) );
  col[0] = static_cast<mwIndex *>( calloc( (n[0]+1),  sizeof(mwIndex)) );

  // ----- find the correct column sizes
  for (mwSize l = 0; l < m[0]; l++){            
    col[0][ col_coo[l]-1 ]++;
  }
  
  for(mwSize i = 0, cumsum = 0; i < n[0]; i++){     
    int temp = col[0][i];
    col[0][i] = cumsum;
    cumsum += temp;
  }
  col[0][n[0]] = m[0];
  
  // ----- copy the row indices to the correct place
  for (mwSize l = 0; l < m[0]; l++){
    int col_l = col_coo[l]-1;
    int dst = col[0][col_l];
    row[0][dst] = row_coo[l]-1;
    
    col[0][ col_l ]++;
  }
  
  // ----- revert the column pointers
  for(mwSize i = 0, last = 0; i < n[0]; i++) {     
    int temp = col[0][i];
    col[0][i] = last;

    last = temp;
  }

  // ~~~~~~~~~~ deallocate memory
  free( row_coo );
  free( col_coo );

}

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

  compute_square(row, col, n, m, d[4]);

  FILE* f;

  f = fopen("raw_results.txt", "w");
  for(mwSize i=0;i<n;i++){
    d[2][i] = d[2][i] -  2 * d[4][i];
    d[3][i] -= d[4][i];
    fprintf(f, "%.1f %.1f %.1f %.1f %.1f\n", d[0][i], d[1][i], d[2][i], d[3][i], d[4][i]);
  }

  fclose(f);
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
  
  readMTX( &row, &col, &n, &m, filename.c_str() );

  printf("number of columns = %ld\n", n);
  printf("non zero elements = %ld\n", m);
  //printArray(col, n + 1);

  //printArray(row, m);
  int graphletSize = 5;
  double **d  = (double **) malloc(graphletSize * sizeof(double *));

  for (int igraph = 0; igraph < graphletSize; igraph++){
    d[igraph] = (double *) calloc(n, sizeof(double));
  }

  struct timeval startwtime, endwtime;
  double duration;

  gettimeofday (&startwtime, NULL);
  computeRaw(row, col, n, m, d);
  gettimeofday (&endwtime, NULL);
  duration = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("[computeRaw took %.4f seconds]\n", duration);

  for (int igraph = 0; igraph < graphletSize; igraph++){
    free(d[igraph]);
  }
  free(d);
  free(row);
  free(col);
  
}