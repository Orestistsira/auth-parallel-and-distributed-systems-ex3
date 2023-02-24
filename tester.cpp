#include <iostream>
#include <fstream>

int main(int argc, char **argv){
    std::cout << "Testing \"raw_results.txt\" and \"raw_results_cuda.txt\"..." << std::endl;

    std::ifstream fSeq("raw_results.txt");
    std::ifstream fCuda("raw_results_cuda.txt");

    if(fSeq.fail()){
        std::cerr << "File " << "raw_results.txt" << " could not be opened! Aborting..." << std::endl;
        exit(1);
    }

    if(fCuda.fail()){
        std::cerr << "File " << "raw_results_cuda.txt" << " could not be opened! Aborting..." << std::endl;
        exit(1);
    }

    double seq, cuda;

    while(fSeq && fCuda){
        fSeq >> seq;
        fCuda >> cuda;

        if(seq != cuda){
            std::cout << "[Error: files are not the same!]" << std::endl;
            exit(1);
        }
    }

    std::cout << "[Correct result! Output files are the same]" << std::endl;

    return 0;
}