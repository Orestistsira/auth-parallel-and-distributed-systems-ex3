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

void printResults(double **d, mwSize n, const char* filename){
  FILE* f;
  f = fopen(filename, "w");

  for(mwSize i = 0; i < n; i++){
    d[2][i] = d[2][i] -  2 * d[4][i];
    d[3][i] -= d[4][i];
    fprintf(f, "%.1f %.1f %.1f %.1f %.1f\n", d[0][i], d[1][i], d[2][i], d[3][i], d[4][i]);
  }
  fclose(f);
}