#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <string>

#include "densematgen.h"

#define VARR_MSG 11
#define RIDXARR_MSG 12
#define CPTRARR_MSG 13
#define OFFSET_MSG 14

using namespace std;

// Based on https://stackoverflow.com/a/868894
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end)
  {
    return *itr;
  }
  return 0;
}

// Based on https://stackoverflow.com/a/868894
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
  return std::find(begin, end, option) != end;
}

class CSRMatrix {
public:
    CSRMatrix(string filePath) {
      ifstream infile(filePath);
      assert(infile.good());
      infile >> rowsNum >> colsNum >> totalNNZ >> maxRowNNZ;
      assert(rowsNum == colsNum);
      V.reserve(totalNNZ);
      Cidx.reserve(totalNNZ);
      Rptr.reserve(rowsNum + 1);
      for (int i = 0; i < totalNNZ; i++) {
        double tmp;
        infile >> tmp;
        V.push_back(tmp);
      }
      for (int i = 0; i < rowsNum + 1; i++) {
        int tmp;
        infile >> tmp;
        Rptr.push_back(tmp);
      }
      for (int i = 0; i < totalNNZ; i++) {
        int tmp;
        infile >> tmp;
        Cidx.push_back(tmp);
      }
      //assert(infile.eof());
      infile.close();
    }

    int Size() {
      return rowsNum;
    }

    void EnsureSizeDivisibleBy(int numProcessors) {
      if (rowsNum % numProcessors == 0) {
        return;
      }

      int newSize = rowsNum + (numProcessors - rowsNum % numProcessors);
      Rptr.resize(newSize + 1, Rptr[rowsNum]);
      rowsNum = colsNum = newSize;
    }

//    vector<vector<int>> rowSplitToProcessors(int numProcessors) {
//      int rowPerProc = rowsNum / numProcessors; // TODO: add zeros when c % p != 0
//      vector<vector<int>> partitions(numProcessors);
//
//      for (int i = 0, currentRow = 0; i < numProcessors; i++, currentRow += rowPerProc) {
//        for (int j = currentRow; j < currentRow + rowPerProc; j++) {
//          int rowBegin = R[j];
//          int rowEnd = R[j + 1];
//          for (int k = rowBegin; k < rowEnd; k++) {
//            partitions[i].push_back(C[k]);
//            partitions[i].push_back(V[k]);
//          }
//        }
//      }
//    }

private:
    vector<double> V;
    vector<int> Cidx;
    vector<int> Rptr;
    int rowsNum, colsNum, totalNNZ, maxRowNNZ;
    friend class CSCMatrix;
};

class CSCMatrix {
public:
    // Based on
    // https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L376
    CSCMatrix(CSRMatrix A) {
      colsNum = A.colsNum;
      rowsNum = A.rowsNum;
      totalNNZ = A.totalNNZ;
      Cptr = vector<int>(colsNum + 1, 0);
      Ridx = vector<int>(totalNNZ, 0);
      V = vector<double>(totalNNZ, 0);

      // Compute number of non-zero entries per column of A
      for (int n = 0; n < totalNNZ; n++) {
        Cptr[A.Cidx[n]]++;
      }

      // Cumsum the nnz per column to get Cptr
      for (int col = 0, cumsum = 0; col < colsNum; col++) {
        int temp  = Cptr[col];
        Cptr[col] = cumsum;
        cumsum += temp;
      }
      Cptr[colsNum] = totalNNZ;

      // Fill Ridx and V
      for (int row = 0; row < rowsNum; row++) {
        for(int jj = A.Rptr[row]; jj < A.Rptr[row+1]; jj++){
          int col  = A.Cidx[jj];
          int dest = Cptr[col];

          Ridx[dest] = row;
          V[dest] = A.V[jj];

          Cptr[col]++;
        }
      }

      for (int col = 0, last = 0; col <= colsNum; col++) {
        int temp  = Cptr[col];
        Cptr[col] = last;
        last = temp;
      }
    }

    vector<MPI_Request> SplitSendToProcessors(int numProcessors) {
      int colPerProc = colsNum / numProcessors; // TODO: add zeros when c % p != 0
      vector<MPI_Request> requests(4 * numProcessors, MPI_REQUEST_NULL);

      for (int i = 0, proc = 0; i < colsNum; i += colPerProc, proc++) {
        int startC = i, endC = i + colPerProc;
        int startRV = Cptr[startC], endRV = Cptr[endC]-1;
        int lenC = endC - startC + 1, lenRV = endRV - startRV + 1;
        int offset = colPerProc * proc;
        MPI_Isend(
                &V[startRV], // buffer
                lenRV, // count
                MPI_DOUBLE, // datatype
                proc, // destination
                VARR_MSG, // tag
                MPI_COMM_WORLD, // communicator
                &requests[proc] // place to store request handle
        );
        MPI_Isend(
                &Ridx[startRV], // buffer
                lenRV, // count
                MPI_INT, // datatype
                proc, // destination
                RIDXARR_MSG, // tag
                MPI_COMM_WORLD, // communicator
                &requests[numProcessors + proc] // place to store request handle
        );
        MPI_Isend(
                &Cptr[startC], // buffer
                lenC, // count
                MPI_INT, // datatype
                proc, // destination
                CPTRARR_MSG, // tag
                MPI_COMM_WORLD, // communicator
                &requests[2 * numProcessors + proc] // place to store request handle
        );
        cerr << "SENDING OFFSET: " << offset << " TO PROC: " << proc << endl;
        MPI_Isend(
                &offset, // buffer
                1, // count
                MPI_INT, // datatype
                proc, // destination
                OFFSET_MSG, // tag
                MPI_COMM_WORLD, // communicator
                &requests[3 * numProcessors + proc] // place to store request handle
        );
      }

      return requests;
      //MPI_Waitall(3 * numProcessors, &requests[0], MPI_STATUSES_IGNORE);
    }

    void Print() {
      cerr << "<SparseCSC> msize=" << rowsNum << endl;
      cerr << "V: ";
      for (auto x : V) cerr << x << " ";
      cerr << endl << "Cptr: ";
      for (auto x : Cptr) cerr << x << " ";
      cerr << endl << "Ridx: ";
      for (auto x : Ridx) cerr << x << " ";
      cerr << endl;
    }

private:
    vector<double> V;
    vector<int> Cptr;
    vector<int> Ridx;
    int rowsNum, colsNum, totalNNZ, maxRowNNZ;
};

class PartialDenseMatrix {
public:
  vector<vector<double>> data;

  PartialDenseMatrix(int rowsNum_, int colsNum_, int offset_) {
    rowsNum = rowsNum_;
    colsNum = colsNum_;
    data = vector<vector<double>>(rowsNum);
    for (int i = 0; i < rowsNum; i++) {
      data[i] = vector<double>(colsNum, 0);
    }
    offset = offset_;
  }

  PartialDenseMatrix& operator=(PartialDenseMatrix&& other) {
    rowsNum = other.rowsNum;
    colsNum = other.colsNum;
    offset = other.colsNum;
    data = vector<vector<double>>(rowsNum);
    for (int i = 0; i < rowsNum; i++) {
      data[i] = move(other.data[i]);
    }
    return *this;
  }

  void FillFromGenerator(int seed, int originalSize) {
    for (int i = 0; i < data.size(); i++) {
      for (int j = 0; j < data[0].size(); j++) {
        if (i < originalSize && j + offset < originalSize) {
          data[i][j] = generate_double(seed, i, offset + j);
        }
        else {
          data[i][j] = 0;
        }
      }
    }
  }

  void Print() {
    cerr << "<PartialDenseMatrix>, offset=" << offset << endl << "Data:" << endl;
    for (int i = 0; i < data.size(); i++) {
      for (int j = 0; j < data[0].size(); j++) {
        cerr << data[i][j] << "  ";
      }
      cerr << endl;
    }
  }

  int HowManyGreaterEqualThan(double value, int originalSize) {
    assert(originalSize <= data.size());
    int result = 0;
    for (int i = 0; i < originalSize; i++) {
      for (int j = 0; j < data[0].size() && j + offset < originalSize; j++) {
        if (data[i][j] >= value) {
          result++;
        }
      }
    }
    return result;
  }

private:
  int offset;
  int rowsNum;
  int colsNum;
};

class PartialCSCMatrix {
public:
    PartialCSCMatrix() {}
    PartialCSCMatrix(int matrixSize_, int numProcessors) {
      MPI_Status status;
      int count;
      matrixSize = matrixSize_;

      for (int i = 0; i < 4; i++) {
        MPI_Probe(
                MPI_ANY_SOURCE, // source
                MPI_ANY_TAG, // tag
                MPI_COMM_WORLD,
                &status
        );

        if (status.MPI_TAG == VARR_MSG) {
          MPI_Get_count(&status, MPI_DOUBLE, &count);
          V = vector<double>(count);
          MPI_Recv(
                  &V[0], // buffer
                  count, // count
                  MPI_DOUBLE, // datatype
                  status.MPI_SOURCE, // source
                  status.MPI_TAG, // tag
                  MPI_COMM_WORLD, // communicator
                  MPI_STATUS_IGNORE // place for status
          );
        } else if (status.MPI_TAG == RIDXARR_MSG) {
          MPI_Get_count(&status, MPI_INT, &count);
          Ridx = vector<int>(count);
          MPI_Recv(
                  &Ridx[0], // buffer
                  count, // count
                  MPI_INT, // datatype
                  status.MPI_SOURCE, // source
                  status.MPI_TAG, // tag
                  MPI_COMM_WORLD, // communicator
                  MPI_STATUS_IGNORE // place for status
          );
        } else if (status.MPI_TAG == CPTRARR_MSG) {
          MPI_Get_count(&status, MPI_INT, &count);
          Cptr = vector<int>(count);
          MPI_Recv(
                  &Cptr[0], // buffer
                  count, // count
                  MPI_INT, // datatype
                  status.MPI_SOURCE, // source
                  status.MPI_TAG, // tag
                  MPI_COMM_WORLD, // communicator
                  MPI_STATUS_IGNORE // place for status
          );
        } else if (status.MPI_TAG == OFFSET_MSG) {
          MPI_Recv(
                  &offset, // buffer
                  1, // count
                  MPI_INT, // datatype
                  status.MPI_SOURCE, // source
                  status.MPI_TAG, // tag
                  MPI_COMM_WORLD, // communicator
                  MPI_STATUS_IGNORE // place for status
          );
          cerr << "RECEIVED OFFSET: " << offset << endl;
        } else {
          assert(false);
        }
      }

      int move_by = Cptr[0];
      for (int i = 0; i < Cptr.size(); i++) {
        Cptr[i] -= move_by;
      }
    }

    vector<MPI_Request> SendTo(int target) {
      vector<MPI_Request> requests(4, MPI_REQUEST_NULL);
      MPI_Isend(
              &V[0], // buffer
              V.size(), // count
              MPI_DOUBLE, // datatype
              target, // destination
              VARR_MSG, // tag
              MPI_COMM_WORLD, // communicator
              &requests[0] // place to store request handle
      );
      MPI_Isend(
              &Ridx[0], // buffer
              Ridx.size(), // count
              MPI_INT, // datatype
              target, // destination
              RIDXARR_MSG, // tag
              MPI_COMM_WORLD, // communicator
              &requests[1] // place to store request handle
      );
      MPI_Isend(
              &Cptr[0], // buffer
              Cptr.size(), // count
              MPI_INT, // datatype
              target, // destination
              CPTRARR_MSG, // tag
              MPI_COMM_WORLD, // communicator
              &requests[2] // place to store request handle
      );
      cerr << "SENDING OFFSET: " << offset << " TO PROC: " << target << endl;
      MPI_Isend(
              &offset, // buffer
              1, // count
              MPI_INT, // datatype
              target, // destination
              OFFSET_MSG, // tag
              MPI_COMM_WORLD, // communicator
              &requests[3] // place to store request handle
      );
      return requests;
    }

    void MultiplyStep(PartialDenseMatrix &B, PartialDenseMatrix &result) {
      for (int result_column = 0; result_column < B.data[0].size(); result_column++) {
        for (int current_column = 0; current_column < Cptr.size() - 1; current_column++) {
          for (int Ridx_pos = Cptr[current_column]; Ridx_pos < Cptr[current_column + 1]; Ridx_pos++) {
            int actual_row = Ridx[Ridx_pos];
            //cerr << "MultiplyStep: actual_row; current_column; Ridx_pos; current_column+offset ";
            //cerr << actual_row << " " << current_column << " " << Ridx_pos << " " << current_column + offset << endl;
            result.data[actual_row][result_column] += V[Ridx_pos] * B.data[current_column + offset][result_column];
          }
        }
      }
    }

    void Print() {
      cerr << "<PartialSparseCSC> msize=" << matrixSize << " offset=" << offset << endl;
      cerr << "V: ";
      for (auto x : V) cerr << x << " ";
      cerr << endl << "Cptr: ";
      for (auto x : Cptr) cerr << x << " ";
      cerr << endl << "Ridx: ";
      for (auto x : Ridx) cerr << x << " ";
      cerr << endl;
    }

private:
    vector<double> V;
    vector<int> Cptr;
    vector<int> Ridx;
    int matrixSize, offset; // colsNum powininen byc tyle ile kawalek, nie cala macierz
};

bool parseCommandLineArgs(int argc, char* argv[], char **csrFilename, bool *inner,
                          bool *print, bool *use_ge, double *ge_value, int *repl_group, int *exponent, int *seed) {
  *csrFilename = getCmdOption(argv, argv + argc, "-f");
  *inner = cmdOptionExists(argv, argv + argc, "-i");
  *print = cmdOptionExists(argv, argv + argc, "-v");
  char *repl_group_string = getCmdOption(argv, argv + argc, "-c");
  char *exponent_string = getCmdOption(argv, argv + argc, "-e");
  char *ge_string = getCmdOption(argv, argv + argc, "-g");
  char *seed_string = getCmdOption(argv, argv + argc, "-s");

  if (*csrFilename == 0 || repl_group_string == 0 || exponent_string == 0 || seed_string == 0) {
    return false;
  }
  *repl_group = atoi(repl_group_string);
  *exponent = atoi(exponent_string);
  *seed = atoi(seed_string);
  if (ge_string != 0) {
    *use_ge = true;
    *ge_value = atof(ge_string);
  } else {
    *use_ge = false;
  }

  return true;
}

void gatherPrintResults(int myRank, int numProcesses, int matrixSize, int originalSize, int colPerProc, PartialDenseMatrix& myDensePart_C) {
  double *sendBuffer = new double[matrixSize * colPerProc];
  double *fullResult;
  for (int i = 0; i < matrixSize; i++) {
    memcpy(sendBuffer + i * colPerProc, &myDensePart_C.data[i][0], colPerProc * sizeof(double));
  }

  if (myRank == 0) {
    fullResult = new double[matrixSize * matrixSize];
  }

  MPI_Gather(
          sendBuffer,
          matrixSize * colPerProc,
          MPI_DOUBLE,
          fullResult,
          matrixSize * colPerProc,
          MPI_DOUBLE,
          0,
          MPI_COMM_WORLD
  );

  delete[] sendBuffer;

  if (myRank == 0) {
    cerr << "Receive whole result and print:" << endl;
    cerr << originalSize << " " << originalSize << endl;
    for (int i = 0; i < originalSize; i++) {
      for (int j = 0; j < numProcesses; j++) {
        for (int k = 0; k < colPerProc; k++) {
          if (j * colPerProc + k < originalSize) {
            cerr << fullResult[j * (matrixSize * colPerProc) + i * colPerProc + k] << " ";
          }
        }
      }
      cerr << endl;
    }
    delete[] fullResult;
  }
}

void reducePrintResult(int myRank, double ge_value, int originalSize, PartialDenseMatrix& myDensePart_C) {
  int myResult = myDensePart_C.HowManyGreaterEqualThan(ge_value, originalSize);
  int fullResult = 0;
  MPI_Reduce(&myResult, &fullResult, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (myRank == 0) {
    cerr << fullResult << endl;
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv); /* intialize the library with parameters caught by the runtime */

  int numProcesses, myRank, matrixSize, originalSize;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  std::ofstream out("out.txt" + to_string(myRank));
  std::streambuf *coutbuf = std::cerr.rdbuf(); //save old buf
  std::cerr.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!

  char *csrFilename;
  bool inner, print, use_ge;
  double ge_value;
  int repl_group, exponent, seed;
  if (!parseCommandLineArgs(
          argc, argv, &csrFilename, &inner, &print, &use_ge, &ge_value, &repl_group, &exponent, &seed)) {
    cerr << "Incorrect command line arguments" << endl;
    return 1;
  }

  cerr << "Arguments: csrFilename=" << csrFilename << " inner=" << inner << " print=" << print << " use_ge=" << use_ge;
  if (use_ge) cerr << " ge_value=" << ge_value;
  cerr << " repl_group=" << repl_group << " exponent=" << exponent << " seed=" << seed << endl;

  cerr << "numProc: " << numProcesses << " myRank: " << myRank << endl;
  int colPerProc;
  int myOffset;

  PartialCSCMatrix mySparsePart_A;

  if (myRank == 0) {
    CSRMatrix csr(csrFilename);
    originalSize = csr.Size();
    csr.EnsureSizeDivisibleBy(numProcesses);
    matrixSize = csr.Size();
    CSCMatrix csc(csr);
    csc.Print();

    colPerProc = matrixSize / numProcesses;
    myOffset = colPerProc * myRank;
    MPI_Bcast(&matrixSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&originalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<MPI_Request> requests = csc.SplitSendToProcessors(numProcesses);
    mySparsePart_A = PartialCSCMatrix(matrixSize, numProcesses);
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  } else {
    MPI_Bcast(&matrixSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&originalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    colPerProc = matrixSize / numProcesses;
    myOffset = colPerProc * myRank;
    mySparsePart_A = PartialCSCMatrix(matrixSize, numProcesses);
  }
  cerr << "Sparse matrix ready." << endl;
  PartialDenseMatrix myDensePart_B(matrixSize, colPerProc, myOffset);
  PartialDenseMatrix myDensePart_C(matrixSize, colPerProc, myOffset);
  myDensePart_B.FillFromGenerator(seed, originalSize);
  MPI_Barrier(MPI_COMM_WORLD);

  cerr << "Matrices ready. A: " << endl;
  mySparsePart_A.Print();
  myDensePart_B.Print();
  //myDensePart_C.Print();
  for (int iter = 0; iter < exponent; iter++) {
    int num_rounds = numProcesses;
    for (int round = 0; round < num_rounds; round++) {
      cerr << "Start round " << round << ". Start multiply step." << endl;
      mySparsePart_A.MultiplyStep(myDensePart_B, myDensePart_C);
      cerr << "Multiply success. Shift." << endl;
      vector <MPI_Request> requests = mySparsePart_A.SendTo((myRank + 1) % numProcesses);
      cerr << "Shift initiated. Receive." << endl;
      mySparsePart_A = PartialCSCMatrix(matrixSize, numProcesses);
      cerr << "Receive finished. Waitall." << endl;
      MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
      cerr << "Round " << round << " finished. New A after shift: " << endl;
      mySparsePart_A.Print();
    }
    if (iter < exponent - 1) { // not last iteration
      myDensePart_B = move(myDensePart_C);
      myDensePart_C = PartialDenseMatrix(matrixSize, colPerProc, myOffset);
    }
  }
  cerr << "Multiply ready. C:" << endl;
  myDensePart_C.Print();

  if (print) {
    gatherPrintResults(myRank, numProcesses, matrixSize, originalSize, colPerProc, myDensePart_C);
  } else if (use_ge) {
    reducePrintResult(myRank, ge_value, originalSize, myDensePart_C);
  }

  std::cerr.rdbuf(coutbuf); //reset to standard output again
  out.close();

  sleep(myRank * 2);
  cerr << "PRINTING RANK " << myRank << endl;
  std::ifstream f("out.txt" + to_string(myRank));
  if (f.is_open())
    std::cerr << f.rdbuf();
  f.close();

  MPI_Finalize(); /* mark that we've finished communicating */
  return 0;
}
