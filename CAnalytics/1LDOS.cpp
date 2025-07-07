#include <iostream>
#include <omp.h>
#include <cmath>
#include <unistd.h>
#include <fstream>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <sstream>
#include <string>
#include <filesystem>

#define THREAD_NUM 32
using namespace std;
namespace fs = std::filesystem;

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
const double a = 0.24595; // nm
const double acc = 0.142;
const float hop = -2.8; // eV
const double Hbar = 6.582119569e-16; // eV * s
const double VFH = VF * Hbar;

const double K0 = (4 * M_PI)/(3*sqrt(3)*acc);

// Locations of Defects
int sep = 0; // nm separation constant
double R1x = -sep * a;
double R1y = 0;
double R2x = sep *a;
double R2y = 0;

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
};

struct vScheme{
    double Kgx;
    double Kgy;
    double Kgtx;
    double Kgty;
    double Kax;
    double Kay;
    double Katx;
    double Katy;
    double Kbx;
    double Kby;
    double Kbtx;
    double Kbty;
};

//const vScheme vs1 = {K0, 0, -K0, 0, K0, 0, -K0, 0, K0, 0, -K0, 0};
vScheme vs1 = {K0, 0, K0, 0, K0, 0, K0, 0, K0, 0, K0, 0};


int updateSep(int sepnum){

    double h = acc * tan(M_PI/3);

    sep = sepnum; // nm separation constant
    R1x = -sep * h;
    R1y = 0;
    R2x = sep *h;
    R2y = 0;

    return 0;
}

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
gsl_complex GF(double w, double Kx, double Ky, double Rx, double Ry, string sindex){

    gsl_complex rval = gsl_complex_rect(0, 0);
    gsl_complex kr_wave = gsl_complex_rect(cos(dot(Kx, Ky, Rx, Ry)), sin(dot(Kx, Ky, Rx, Ry)));
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
            thet_wave = gsl_complex_rect(vI * cos(vI * thet), vI * sin(vI * thet));
        }
        else if (sindex == "BA"){
            thet_wave = gsl_complex_rect(vI * cos(vI * -thet), vI * sin(vI * -thet));
        }

        rval = gsl_complex_mul(gsl_complex_mul(frac, Hankel1(w * normR/VFH)), thet_wave);
    }

    return rval;

}

// This one did not have proper units. i.e smoothed Green's function vs lattice greens funciton.
// Check this expression, not entirely confident
//gsl_complex G0AA(double w){
//    return gsl_complex_rect(w/(sqrt(3)*M_PI*pow(hop, 2)) * log(pow(w, 2)/(sqrt(3)*M_PI*pow(hop, 2))), -abs(w)/(sqrt(3)*pow(hop, 2)));
//}
gsl_complex G0AA(double w){
    //double real_part = (M_PI * w)/(pow(VF*Hbar, 2)) * log(pow(w, 2)/(pow((3/2)*hop, 2)-pow(w, 2)));
    double real_part = (M_PI * w)/(pow(VF*Hbar, 2)) * log(pow(w, 2)/(pow(hop, 2)));
    double imag_part = -(pow(M_PI, 2)*abs(w))/(pow(VF*Hbar, 2));

    return gsl_complex_rect(real_part, imag_part);
}

gsl_complex telem(double w){
    return gsl_complex_div(gsl_complex_rect(V0, 0), gsl_complex_add_real(gsl_complex_mul_real(G0AA(w), -V0), 1));
}

/*
gsl_complex Rfrac(double w, double K1x, double K1y, double K2x, double K2y){
    GFParams p1 = {K1x, K1y, R1x-R2x, R1y-R2y, "AA"};
    GFParams p2 = {K2x, K2y, -(R1x-R2x), -(R1y-R2y), "AA"};

    gsl_complex tnum = telem(0.2);
    gsl_complex tnum2 = gsl_complex_mul(tnum, tnum);

    gsl_complex denom = gsl_complex_add_real(gsl_complex_mul(gsl_complex_negative(tnum2),gsl_complex_mul(GF(w, p1.Kx, p1.Ky, p1.Rx, p1.Ry, p1.sindex),\
    GF(w, p2.Kx, p2.Ky, p2.Rx, p2.Ry, p2.sindex))), 1);

    return gsl_complex_div(gsl_complex_rect(1, 0), denom);
}
*/


gsl_complex orderone(double w, GFParams p1, GFParams p2){
    return gsl_complex_mul(GF(w, p1.Kx, p1.Ky, p1.Rx, p1.Ry, p1.sindex), GF(w, p2.Kx, p2.Ky, p2.Rx, p2.Ry, p2.sindex));
}

gsl_complex ordertwo(double w, GFParams p1, GFParams p2, GFParams p3){
    return gsl_complex_mul(GF(w, p1.Kx, p1.Ky, p1.Rx, p1.Ry, p1.sindex), orderone(w, p2, p3));
}


double f(double w, double x, double y, vScheme vs){
    double C = -2 * pow(w, 2)/(pow(VFH, 4)*16);

    double thet = atan2(y, x);
    double gamma = norm(x, y)*w/VFH;
    double term1 = C*cos(dot(-2*K0, 0, x, y))*GSL_IMAG(gsl_complex_mul(gsl_complex_mul(telem(0.2), Hankel0(gamma)), Hankel0(gamma)));
    double term2 = -C*cos(dot(-2*K0, 0, x, y)-2*thet)*GSL_IMAG(gsl_complex_mul(gsl_complex_mul(telem(0.2), Hankel1(gamma)), Hankel1(gamma)));

    return term1 + term2;
}

int calculateGrid(double* d_list){
    omp_set_num_threads(THREAD_NUM); // set number of threads in "parallel" blocks
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                d_list[j*nx + i] = f(0.2, cFI(i, dx, shift_x), cFI(j, dy, shift_y), vs1);
            }
        }
    }

    return 0;
}

int save2file(double* d_list, string fname){
    string fodir = "output/single" + std::to_string(sep) + "a/";
    fs::create_directories(fodir);
    ofstream myfile (fodir + fname + "-ldos.tsv");
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
    string fname = "onedef";
    calculateGrid(output);
    save2file(output, fname);
    return 0;
}