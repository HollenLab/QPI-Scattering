#include <iostream>
#include <omp.h>
#include <unistd.h>
#include <fstream>
#define THREAD_NUM 32
using namespace std;

static int nx = 101;
static int ny = 101;

static double dx = 1.0;
static double dy = 1.0;

static double shift_x = 50;
static double shift_y = 50;

// Initalize Array
double *output = new double[nx*ny]; 

// coord from Index
double cFI(int i, double dix, double shift){
    return i * dix - shift;
}

double f(double x, double y){
    return x * y;
}

int calculateGrid(double* d_list){
    omp_set_num_threads(THREAD_NUM); // set number of threads in "parallel" blocks
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                d_list[j*nx + i] = f(cFI(i, dx, shift_x), cFI(j, dy, shift_y));
            }
        }
    }

    return 0;
}

int save2file(double* d_list){
    ofstream myfile ("ldos.tsv");
    if (myfile.is_open()){
         // Saving info for plotting
         myfile << "nx" << "\t" << "ny" << "\t" << "dx" << "\t" << "dy" << "\t" << "sx" << "\t" << "sy" << "\n";
         myfile << nx << "\t" << ny << "\t" << dx << "\t" << dy << "\t" << shift_x << "\t" << shift_y << "\n\n";

        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
                if (i == nx - 1){
                    myfile << d_list[j*nx + i];
                }
                else{
                    myfile << d_list[j*nx + i] << "\t";
                }
            }
            myfile << "\n";
        }

        myfile.close();
    }
    else{
        cout << "Unable to open file";
    }


    return 0;
}

int main()
{
    calculateGrid(output);
    save2file(output);
    return 0;
}