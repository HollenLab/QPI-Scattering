#include <iostream>
#include <omp.h>
#include <cmath>
#include <unistd.h>
#include <fstream>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#define THREAD_NUM 32
using namespace std;

////////////////////////////////////////
// Grid Parameters //
/////////////////////

static int nx = 1001;
static int ny = 1001;

static double dx = 0.02;
static double dy = 0.02;

static double shift_x = 10;
static double shift_y = 10;

// Initalize Array
double *output = new double[nx*ny]; 
//////////////////////////////////////////
// Physics Parameters //
const float V0 = 1; // eV
const double VF = 9.060911856897319e14; // nm/s 
const float a = 0.24595; // nm
const int sep = 7; // nm separation constant
const float hop = -2.8; // eV
const double Hbar = 6.582119569e-16; // eV * s
const double VFH = VF * Hbar;

const double K0 = (4 * M_PI)/(3*sqrt(3)*a);

// Locations of Defects
const double R1x = -sep * a;
const double R1y = 0;
const double R2x = sep *a;
const double R2y = 0;

////////////////////
// Misc Parameters//
////////////////////
const double smooth = 1e-9; 


struct GFParams{
    double Kx;
    double Ky;
    double Rx;
    double Ry;
    string sindex;
    bool isForward;
};


// coord from Index
double cFI(int i, double dix, double shift){
    return i * dix - shift;
}

//////////////////////////
// 2D Vector Operations //
//////////////////////////
double dot(double vx, double vy, double ux, double uy){
    return (vx * ux) + (vy * uy);
}

double norm(double vx, double vy){
    return sqrt(pow(vx, 2) + pow(vy, 2));
}

double theta(double vx, double vy){
    return atan2(vy, vx);
}

//////////////////////////////////////////////
// Special Functions and Complex Operations //
//////////////////////////////////////////////
gsl_complex Hankel0(double x){
    if (x < 0){
        cout << "Hankel x less than zero!\n";
    }
    if (x == 0){
        x += 1e-9;
    }
    return gsl_complex_rect(gsl_sf_bessel_J0(x), gsl_sf_bessel_Y0(x));
}
gsl_complex Hankel1(double x){
    if (x < 0){
        cout << "Hankel x less than zero!\n";
    }
    if (x == 0){
        x += 1e-9;
    }
    return gsl_complex_rect(gsl_sf_bessel_J1(x), gsl_sf_bessel_Y1(x));
}
int vIndex(double Kx, double Ky){
    int rval = 0;
    if (Kx > 0){
        rval = 1;
    }
    else {
        rval = -1;
    }

    return rval;
}

//multiply by i trick
gsl_complex mbyI(gsl_complex z){
    return gsl_complex_rect(-GSL_IMAG(z), GSL_REAL(z));
}
// dindex: forward = 1 backward = -1
gsl_complex GF(double w, double Kx, double Ky, double Rx, double Ry, string sindex, bool isForward){

    int dindex = -1;
    if(isForward == true){
        dindex = 1;
    }

    gsl_complex rval = gsl_complex_rect(0, 0);
    gsl_complex kr_wave = gsl_complex_rect(cos(dindex * dot(Kx, Ky, Rx, Ry)), sin(dindex * dot(Kx, Ky, Rx, Ry)));
    gsl_complex frac = gsl_complex_mul_real(kr_wave, (-1 * w)/pow(2 * VFH, 2));

    int vI = vIndex(Kx, Ky);
    double thet = theta(Rx, Ry);

    double normR = norm(Rx, Ry);

    if (sindex == "AA" || sindex == "BB"){
        rval = gsl_complex_mul(frac, mbyI(Hankel0(w * normR/VFH)));
    }
    else if(sindex == "AB" || sindex == "BA"){
        gsl_complex thet_wave;
        if (sindex == "AB"){
            thet_wave = gsl_complex_rect(dindex * vI * cos(vI * thet), dindex * vI * sin(vI * thet));
        }
        else if (sindex == "BA"){
            thet_wave = gsl_complex_rect(dindex * vI * cos(vI * -thet), dindex * vI * sin(vI * -thet));
        }

        rval = gsl_complex_mul(gsl_complex_mul(frac, Hankel1(w * normR/VFH)), thet_wave);
    }

    return rval;

}

// Check this expression, not entirely confident
gsl_complex G0AA(double w){
    return gsl_complex_rect(w/(sqrt(3)*M_PI*pow(hop, 2)) * log(pow(w, 2)/(sqrt(3)*M_PI*pow(hop, 2))), -abs(w)/(sqrt(3)*pow(hop, 2)));
}

gsl_complex telem(double w){
    return gsl_complex_div(gsl_complex_rect(V0, 0), gsl_complex_add_real(gsl_complex_mul_real(G0AA(w), -V0), 1));
}

gsl_complex orderone(double w, GFParams p1, GFParams p2){
    return gsl_complex_mul(gsl_complex_mul(GF(w, p1.Kx, p1.Ky, p1.Rx, p1.Ry, p1.sindex, p1.isForward), telem(w)), GF(w, p2.Kx, p2.Ky, p2.Rx, p2.Ry, p2.sindex, p2.isForward));
}

double f(double x, double y){
    GFParams p1 = {-K0, 0, x-R1x, y-R1y, "AA", true};
    GFParams p2 = {K0, 0, x-R1x, y-R1y, "AA", false};
    gsl_complex term1 = orderone(0.2, p1, p2);

    p1 = {-K0, 0, x-R1x, y-R1y, "BA", true};
    p2 = {K0, 0, x-R1x, y-R1y, "AB", false};
    gsl_complex term2 = orderone(0.2, p1, p2);

    p1 = {-K0, 0, x-R2x, y-R2y, "AA", true};
    p2 = {K0, 0, x-R2x, y-R2y, "AA", false};
    gsl_complex term3 = orderone(0.2, p1, p2);

    p1 = {-K0, 0, x-R2x, y-R2y, "BA", true};
    p2 = {K0, 0, x-R2x, y-R2y, "AB", false};
    gsl_complex term4 = orderone(0.2, p1, p2);

    gsl_complex term_order_one = gsl_complex_add(gsl_complex_add(gsl_complex_add(term4, term3), term2), term1);

    return GSL_IMAG(term_order_one);
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
         myfile << "nx" << "\t" << "ny" << "\t" << "dx" << "\t" << "dy" << "\t" << "sx" << "\t" << "sy" << "\t" << "sep" << "\n";
         myfile << nx << "\t" << ny << "\t" << dx << "\t" << dy << "\t" << shift_x << "\t" << shift_y << "\t" << sep << "\n\n";

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