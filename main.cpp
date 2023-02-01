#include "fglt.hpp"

// void compute_d4(mwIndex *rowA, mwIndex *colA, double *val, mwSize n, mwSize m, mwIndex *rowRes, mwIndex *colRes, double *valRes){
//     int *row_count = (int*)calloc(n, sizeof(int));

//     // Count non-zero elements in each column of C
//     for (int j = 0; j < n; j++) {
//         for (int i = colA[j]; i < colA[j+1]; i++) {
//             int row = rowA[i];
//             for (int k = colA[row]; k < colA[row+1]; k++) {
//                 int col = rowA[k];
//                 row_count[col]++;
//             }
//         }
//     }

//     // Fill col_ptr array
//     colRes[0] = 0;
//     for (int i = 0; i < n; i++) {
//         colRes[i+1] = colRes[i] + row_count[i];
//         row_count[i] = 0;
//     }

//     // Fill row_ind and val arrays
//     for (int j = 0; j < n; j++) {
//         for (int i = colA[j]; i < colA[j+1]; i++) {
//             int row = rowA[i];
//             double a_ij = val[i];
//             for (int k = colA[row]; k < colA[row+1]; k++) {
//                 int col = rowA[k];
//                 double a_ik = val[k];
//                 int index = colRes[col] + row_count[col];
//                 rowRes[index] = j;
//                 valRes[index] += a_ij * a_ik;
//                 row_count[col]++;
//             }
//         }
//     }

//     free(row_count);
// }

//for csr (csc only for symmetric matrices)
void compute_square(mwIndex *row, mwIndex *col, mwSize n, mwSize m, mwIndex *rowRes, mwIndex *colRes, double *valRes){
    int ip = 0;

    double* d = (double*) calloc(n, sizeof(double));

    int *xb = (int*) malloc(n * sizeof(int));
    for(int i = 0; i < n; i++){
        xb[i] = -1;
    }
    int *x = (int*) malloc(n * sizeof(int));

    //for each column (i)
    for(int i = 0; i < n; i++){
        printf("i = %d:\n", i);
        colRes[i] = ip;

        //for each row (j) that column (i) has an element
        for(int jp = col[i]; jp < col[i+1];jp++){
            int j = row[jp];
            printf("j = %d\n", j);

            //for each row (k) that column (j) has an element
            for(int kp = col[j]; kp < col[j+1]; kp++){
                int k = row[kp];
                printf("k = %d\n", k);

                //if(k == 1 && i == 1) continue;
                bool compute = false;
                for(int fp = col[i]; fp < col[i+1]; fp++){
                    int f = row[fp];
                    if(f == k){
                        compute = true;
                        break;
                    } 
                }

                if(!compute) continue;

                if(xb[k] != i){
                    printf("Loaded element (%d, %d)\n", k, i);
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

        for(int vp = colRes[i]; vp < ip; vp++){
            int v = rowRes[vp];

            valRes[vp] = x[v];
            d[i] += x[v];
        }

        printf("\n");
    }
    printf("\n");

    for(int i = 0; i < n; i++){
        printf("%f ", d[i]);
    }
    printf("\n");

    colRes[n] = ip;
}

void printArray(mwIndex* array, mwSize n){
    for(int i=0;i<n;i++){
        printf("%ld\n", array[i]);
    }
    printf("\n");
}

void printArrayD(double* array, mwSize n){
    for(int i=0;i<n;i++){
        printf("%f\n", array[i]);
    }
    printf("\n");
}

const int m = 4;
const int n = 3;

int main(int argc, char** argv){
    mwIndex row[4] = {0, 1, 0, 2};
    mwIndex col[4] = {0, 2, 3, 4};

    double val[4] = {1, 1, 1, 1};

    mwIndex *rowRes = (mwIndex*) malloc(2 * m * sizeof(mwIndex));
    mwIndex *colRes = (mwIndex*) calloc((n+1), sizeof(mwIndex));
    double *valRes = (double*) malloc(2 * m * sizeof(double));

    printf("here\n");

    compute_square(row, col, n, m, rowRes, colRes, valRes);

    int mRes = colRes[n];

    printf("nnz of result = %d\n", mRes);
    printArray(rowRes, mRes);
    printArray(colRes, (n+1));
    printArrayD(valRes, mRes);

    free(rowRes);
    free(colRes);
    free(valRes);

    return 0;
}