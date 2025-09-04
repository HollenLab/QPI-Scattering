#include <iostream>
#include <omp.h>
#include <cmath>
#include <unistd.h>
#include <fstream>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_integration.h>
#include <sstream>
#include <string>
#include <filesystem>

#define THREAD_NUM 32
using namespace std;
namespace fs = std::filesystem;

//////////////////////
// Grid Parameters //
/////////////////////

static int nx = 501;
static int ny = 501;

static double shift_x = 5;
static double shift_y = 5;

static double dx = (2*shift_x)/(nx-1);
static double dy = (2*shift_y)/(ny-1);

// Initalize Array
double *output = new double[nx*ny]; 

////////////////////////
// Physics Parameters //
///////////////////////
double V0 = 20; // eV
const double VF = 906091185689731.9; // nm/s 
const double a = 0.24595; // nm
//const double acc = 0.142; //nm
const double acc = a/sqrt(3);
const double hop = -2.8; // eV
const double Hbar = 6.582119569e-16; // eV * s
const double VFH = VF * Hbar;

const double K0 = (4 * M_PI)/(3*sqrt(3)*acc);

// Rotate by 60 degrees
double K1x = 0.5 * K0;
double K1y = sqrt(3)/2 * K0;

// Rotate by 120 degrees
double K2x = -0.5 * K0;
double K2y = sqrt(3)/2 * K0;

// Locations of Defects
int sep = 20; // nm separation constant
//double R1x = -sep * a;
//double R1y = 0;
//double R2x = sep *a;
//double R2y = 0;

// Hardcode
//double R1x = -1.23;
//double R1y = -0.39;
//double R2x = 1.23;
//double R2y = 0.39;

//double R1x = -0.55;
//double R1y = 0.071;
//double R2x = 0.55;
//double R2y = -0.071;

double R1x = -1.045;
double R1y = 0.745;
double R2x = 1.045;
double R2y = -0.745;

//double R1x = -1.72;
//double R1y = 0.071;
//double R2x = +1.72;
//double R2y = 0.071;

////////////////////
// Misc Parameters//
////////////////////
const double smooth = 0; 


struct GFParams{
    double Kx;
    double Ky;
    double Rx;
    double Ry;
    string sindex;
};

// Change the separation and all the variables related to separation
int updateSep(int sepnum){

    // Separation is an integer related to number of unit cells in between the defect's location
    // Two unit cells are seperated by the width of the graphene hexagon a_cc * tan(60)
    double h = acc * tan(M_PI/3);

    sep = sepnum; // nm separation constant
    R1x = -sep * h;
    R1y = -acc/2;
    R2x = sep *h;
    R2y = -acc/2;

    return 0;
}

// Maps the array's element index to coordinate position value
// The shift determines the grid. i.e shift of 10 gives you a grid defined 
// from -10nm to 10nm i.e a 20x20 nm grid with zero at the center.
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
    return gsl_complex_rect(gsl_sf_bessel_J0(x + smooth), gsl_sf_bessel_Y0(x + smooth));
}
gsl_complex Hankel1(double x){
    if (x < 0){
        cout << "Hankel x less than zero!\n";
    }
    if (x == 0){
        x += 1e-9;
    }
    return gsl_complex_rect(gsl_sf_bessel_J1(x + smooth), gsl_sf_bessel_Y1(x + smooth));
}
// Tells you for a given K vector which valley you are located and assigns either 1 or -1
int vIndex(double Kx, double Ky){
    int rval = 0;

    if (Kx == K0 || (Kx == K2x && Ky == K2y) || (Kx == -K1x && Ky == -K1y)){
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

// 2x2 Green's Function Matrix
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

// Onsite Green's Function Matrix Element r->0
// I think energy cutoff should relate to hopping? Makes sense
// Or the energy value that relates to normalization...
gsl_complex G0AA(double w){
    //double real_part = (w)/(pow(2*VF*Hbar, 2)*4*M_PI) * log(pow(w, 2)/(8*M_PI*pow(VFH, 2) - pow(w, 2)));
    double real_part = (w)/(pow(2*VF*Hbar, 2)*4*M_PI) * log(pow(w, 2)/(pow(hop, 2) - pow(w, 2)));
    double imag_part = -(M_PI*abs(w))/(pow(VF*Hbar, 2)*4*M_PI);

    return gsl_complex_rect(real_part, imag_part);
}
// Single T Matrix Element
gsl_complex telem(double w){
    return gsl_complex_div(gsl_complex_rect(V0, 0), gsl_complex_add_real(gsl_complex_mul_real(G0AA(w), -V0), 1));
}
// Multiplying GF's Together
gsl_complex orderone(double w, GFParams p1, GFParams p2){
    return gsl_complex_mul(GF(w, p1.Kx, p1.Ky, p1.Rx, p1.Ry, p1.sindex), GF(w, p2.Kx, p2.Ky, p2.Rx, p2.Ry, p2.sindex));
}

gsl_complex ordertwo(double w, GFParams p1, GFParams p2, GFParams p3){
    return gsl_complex_mul(GF(w, p1.Kx, p1.Ky, p1.Rx, p1.Ry, p1.sindex), orderone(w, p2, p3));
}
// Condensed Geometric Series Term
gsl_complex Rfrac(double w){

    double d12x = R1x - R2x;
    double d12y = R1y - R2y;

    // Does not depend on K1 or K2 so we hard code them here
    double K1x = K0;
    double K1y = 0;
    double K2x = K0;
    double K2y = 0;


    GFParams p1 = {K1x, K1y, d12x, d12y, "AA"};
    GFParams p2 = {K2x, K2y, -d12x, -d12y, "AA"};

    gsl_complex tnum = telem(w);
    gsl_complex tnum2 = gsl_complex_mul(tnum, tnum);

    gsl_complex add4 = gsl_complex_mul(tnum2, orderone(w, p1, p2));
    gsl_complex denom = gsl_complex_add(gsl_complex_rect(1, 0), gsl_complex_mul(gsl_complex_rect(-1, 0), add4));

    return gsl_complex_div(gsl_complex_rect(1, 0), denom);
    //return gsl_complex_rect(1, 0);
}

gsl_complex RfracAB(double w){

    double d12x = R1x - R2x;
    double d12y = R1y - R2y;

    // Does not depend on K1 or K2 so we hard code them here
    double K1x = K2x;
    double K1y = K2y;
    double K2x = K2x;
    double K2y = K2y;


    GFParams p1 = {K1x, K1y, d12x, d12y, "AB"};
    GFParams p2 = {K2x, K2y, -d12x, -d12y, "BA"};

    gsl_complex tnum = telem(w);
    gsl_complex tnum2 = gsl_complex_mul(tnum, tnum);

    gsl_complex add4 = gsl_complex_mul(tnum2, orderone(w, p1, p2));
    gsl_complex denom = gsl_complex_add(gsl_complex_rect(1, 0), gsl_complex_mul(gsl_complex_rect(-1, 0), add4));

    return gsl_complex_div(gsl_complex_rect(1, 0), denom);
    //return gsl_complex_rect(1, 0);
}

gsl_complex RfracBA(double w){

    double d12x = R1x - R2x;
    double d12y = R1y - R2y;

    // Does not depend on K1 or K2 so we hard code them here
    double K1x = K2x;
    double K1y = K2y;
    double K2x = K2x;
    double K2y = K2y;


    GFParams p1 = {K1x, K1y, d12x, d12y, "BA"};
    GFParams p2 = {K2x, K2y, -d12x, -d12y, "AB"};

    gsl_complex tnum = telem(w);
    gsl_complex tnum2 = gsl_complex_mul(tnum, tnum);

    gsl_complex add4 = gsl_complex_mul(tnum2, orderone(w, p1, p2));
    gsl_complex denom = gsl_complex_add(gsl_complex_rect(1, 0), gsl_complex_mul(gsl_complex_rect(-1, 0), add4));

    return gsl_complex_div(gsl_complex_rect(1, 0), denom);
    //return gsl_complex_rect(1, 0);
}



struct ldos_param_c{
    double x;
    double y;
    double dKx;
    double dKy;
    string sblattices;
};

double gma(double w, double vx, double vy){
    return w * norm(vx, vy)/VFH;
}

// Condensed LDOS
// Precaculate t and R for given energy
// Change valley scheme to not
double rho(int index, double w, double x, double y){

    double dR1x = x - R1x;
    double dR1y = y - R1y;
    double dR2x = x - R2x;
    double dR2y = y - R2y;
    double d12x = R1x - R2x;
    double d12y = R1y - R2y;


    // Can figure out how to make these shared
    // Does the valley actually matter for R? it shouldnt...
    gsl_complex R = Rfrac(w);
    gsl_complex t = telem(w);
    gsl_complex t2 = gsl_complex_mul(t, t);

    // 5-8 are multiplied by i so take minus of the real

    double result = 0;
    // 2 comes from euler cosine identity
    double cfrac2 = pow(w, 2)/pow(2*VFH, 4);
    double cfrac3 = pow(w, 3)/pow(2*VFH, 6);
    if (index == 1){
        gsl_complex h01r = Hankel0(gma(w, dR1x, dR1y));
        
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h01r, h01r))));
    }
    else if (index == 2){
        gsl_complex h11r = Hankel1(gma(w, dR1x, dR1y));
        
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h11r, h11r))));
    }
    else if(index == 3){
        gsl_complex h02r = Hankel0(gma(w, dR2x, dR2y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h02r, h02r))));
    }
    else if (index == 4){
        gsl_complex h12r = Hankel1(gma(w, dR2x, dR2y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h12r, h12r))));
    }
    else if (index == 5){
        gsl_complex hankels = gsl_complex_mul(gsl_complex_mul(Hankel0(gma(w, d12x, d12y)), Hankel0(gma(w, dR1x, dR1y))), Hankel0(gma(w, dR2x, dR2y)));
        gsl_complex cpref = gsl_complex_mul(R, t2);

        return cfrac3 * GSL_REAL(gsl_complex_mul(cpref, hankels));
    }
    else if (index == 6){
        gsl_complex hankels = gsl_complex_mul(gsl_complex_mul(Hankel0(gma(w, d12x, d12y)), Hankel1(gma(w, dR1x, dR1y))), Hankel1(gma(w, dR2x, dR2y)));
        gsl_complex cpref = gsl_complex_mul(R, t2);

        return cfrac3 * GSL_REAL(gsl_complex_mul(cpref, hankels));
    }
    else{
        return 0;
    }
}

double rhoAB(int index, double w, double x, double y){
    double dR1x = x - R1x;
    double dR1y = y - R1y;
    double dR2x = x - R2x;
    double dR2y = y - R2y;
    double d12x = R1x - R2x;
    double d12y = R1y - R2y;


    // Can figure out how to make these shared
    // Does the valley actually matter for R? it shouldnt...
    gsl_complex R = RfracAB(w);
    gsl_complex t = telem(w);
    gsl_complex t2 = gsl_complex_mul(t, t);

    double cfrac2 = pow(w, 2)/pow(2*VFH, 4);
    double cfrac3 = pow(w, 3)/pow(2*VFH, 6);

    if (index == 1){
        gsl_complex h01r = Hankel0(gma(w, dR1x, dR1y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h01r, h01r)))); 
    }
    else if (index == 2){
        gsl_complex h11r = Hankel1(gma(w, dR1x, dR1y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h11r, h11r)))); 
    }
    else if (index == 3){
        gsl_complex h02r = Hankel0(gma(w, dR2x, dR2y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h02r, h02r))));
    }
    else if (index == 4){
        gsl_complex h12r = Hankel1(gma(w, dR2x, dR2y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h12r, h12r))));
    }
    else if (index == 5){
        gsl_complex hankels = gsl_complex_mul(gsl_complex_mul(Hankel1(gma(w, d12x, d12y)), Hankel0(gma(w, dR1x, dR1y))), Hankel1(gma(w, dR2x, dR2y)));
        gsl_complex cpref = gsl_complex_mul(R, t2);

        return cfrac3 * GSL_REAL(gsl_complex_mul(cpref, hankels));
    }
    else if (index == 6){
        gsl_complex hankels = gsl_complex_mul(gsl_complex_mul(Hankel1(gma(w, d12x, d12y)), Hankel1(gma(w, dR1x, dR1y))), Hankel0(gma(w, dR2x, dR2y)));
        gsl_complex cpref = gsl_complex_mul(R, t2);

        return cfrac3 * GSL_REAL(gsl_complex_mul(cpref, hankels));
    }
    else{
        return 0;
    }

}

double rhoBA(int index, double w, double x, double y){
    double dR1x = x - R1x;
    double dR1y = y - R1y;
    double dR2x = x - R2x;
    double dR2y = y - R2y;
    double d12x = R1x - R2x;
    double d12y = R1y - R2y;


    // Can figure out how to make these shared
    // Does the valley actually matter for R? it shouldnt...
    gsl_complex R = RfracBA(w);
    gsl_complex t = telem(w);
    gsl_complex t2 = gsl_complex_mul(t, t);

    double cfrac2 = pow(w, 2)/pow(2*VFH, 4);
    double cfrac3 = pow(w, 3)/pow(2*VFH, 6);

    if (index == 1){
        gsl_complex h01r = Hankel0(gma(w, dR1x, dR1y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h01r, h01r)))); 
    }
    else if (index == 2){
        gsl_complex h11r = Hankel1(gma(w, dR1x, dR1y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h11r, h11r)))); 
    }
    else if (index == 3){
        gsl_complex h02r = Hankel0(gma(w, dR2x, dR2y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h02r, h02r))));
    }
    else if (index == 4){
        gsl_complex h12r = Hankel1(gma(w, dR2x, dR2y));
        
        return cfrac2 * GSL_IMAG(gsl_complex_mul(t, gsl_complex_mul(R, gsl_complex_mul(h12r, h12r))));
    }
    else if (index == 5){
        gsl_complex hankels = gsl_complex_mul(gsl_complex_mul(Hankel1(gma(w, d12x, d12y)), Hankel0(gma(w, dR1x, dR1y))), Hankel1(gma(w, dR2x, dR2y)));
        gsl_complex cpref = gsl_complex_mul(R, t2);

        return cfrac3 * GSL_REAL(gsl_complex_mul(cpref, hankels));
    }
    else if (index == 6){
        gsl_complex hankels = gsl_complex_mul(gsl_complex_mul(Hankel1(gma(w, d12x, d12y)), Hankel1(gma(w, dR1x, dR1y))), Hankel0(gma(w, dR2x, dR2y)));
        gsl_complex cpref = gsl_complex_mul(R, t2);

        return cfrac3 * GSL_REAL(gsl_complex_mul(cpref, hankels));
    }
    else{
        return 0;
    }

}

double condensedLDOS(double w, double x, double y, double dKx, double dKy, string sblattices){
    double dR1x = x - R1x;
    double dR1y = y - R1y;
    double dR2x = x - R2x;
    double dR2y = y - R2y;
    double d12x = R1x - R2x;
    double d12y = R1y - R2y;
     

    double thet1 = theta(dR1x, dR1y);
    double thet2 = theta(dR2x, dR2y); //add minus to change sublattice?

    if (sblattices == "AA"){
        return (rho(1, w, x, y) - rho(5, w, x, y))*cos(dot(dKx, dKy, dR1x, dR1y))+(rho(3, w, x, y) - rho(5, w, x, y))*cos(dot(dKx, dKy, dR2x, dR2y))\
    - rho(2, w, x, y)*cos(dot(dKx, dKy, dR1x, dR1y)-2*thet1) - rho(4, w, x, y)*cos(dot(dKx, dKy, dR2x, dR2y)-2*thet2) \
    + rho(6, w, x, y)*(cos(dot(dKx, dKy, dR1x, dR1y)-thet1-thet2)+cos(dot(dKx, dKy, dR2x, dR2y)-thet1-thet2));
    }
    else if (sblattices == "AB"){
        return rhoAB(1, w, x, y) * cos(dot(dKx, dKy, dR1x, dR1y)) - rhoAB(2, w, x, y) * cos(dot(dKx, dKy, dR1x, dR1y)-2*thet1) \
        + rhoAB(3, w, x, y) * cos(dot(dKx, dKy, dR2x, dR2y)) - rhoAB(4, w, x, y) * cos(dot(dKx, dKy, dR2x, dR2y)+2*thet2) \
        - 2*rhoAB(5, w, x, y) * cos(dot(dKx, dKy, dR1x, dR1y) + thet2) - 2*rhoAB(6, w, x, y) * cos(dot(dKx, dKy, dR1x, dR1y)- thet1) \
        + 2*rhoAB(5, w, x, y) * cos(dot(dKx, dKy, dR2x, dR2y) + thet2) + 2*rhoAB(6, w, x, y) * cos(dot(dKx, dKy, dR2x, dR2y)- thet1);
    }
    // This is just a guess
    else if (sblattices == "BA"){
        return (rhoBA(1, w, x, y) * cos(dot(dKx, dKy, dR1x, dR1y)) - rhoBA(2, w, x, y) * cos(dot(dKx, dKy, dR1x, dR1y)+2*thet1) \
        + rhoBA(3, w, x, y) * cos(dot(dKx, dKy, dR2x, dR2y)) - rhoBA(4, w, x, y) * cos(dot(dKx, dKy, dR2x, dR2y)-2*thet2) \
        - 2*rhoBA(5, w, x, y) * cos(dot(dKx, dKy, dR1x, dR1y) - thet2) - 2*rhoBA(6, w, x, y) * cos(dot(dKx, dKy, dR1x, dR1y)+ thet1) \
        + 2*rhoBA(5, w, x, y) * cos(dot(dKx, dKy, dR2x, dR2y) - thet2) + 2*rhoBA(6, w, x, y) * cos(dot(dKx, dKy, dR2x, dR2y)+ thet1));
    }
    // This is just a guess
    else if (sblattices == "BB"){
        return (rho(1, w, x, y) - rho(5, w, x, y))*cos(dot(dKx, dKy, dR1x, dR1y))+(rho(3, w, x, y) - rho(5, w, x, y))*cos(dot(dKx, dKy, dR2x, dR2y))\
    + rho(2, w, x, y)*cos(dot(dKx, dKy, dR1x, dR1y)+2*thet1) + rho(4, w, x, y)*cos(dot(dKx, dKy, dR2x, dR2y)+2*thet2) \
    - rho(6, w, x, y)*(cos(dot(dKx, dKy, dR1x, dR1y)+thet1+thet2)+cos(dot(dKx, dKy, dR2x, dR2y)+thet1+thet2));
    }
    else{
        return 0;
    }
    
}

struct ldos_param{
    double x;
    double y;
    double dKx;
    double dKy;
    string sblattices;
};

double f_wrapper_cond(double w, void* params) {
    ldos_param_c* p = static_cast<ldos_param_c*>(params);

    return condensedLDOS(w, p->x, p->y, p->dKx, p-> dKy, p-> sblattices);
}


double integrate_f_cond(double x, double y){
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
        
    double result, error;

    //ldos_param lp = {x, y, K0, 0};
    ldos_param lp = {x, y, K1x, K1y, "AA"};

    gsl_function F;
    F.function = &f_wrapper_cond;
    F.params = &lp;
    
    gsl_integration_qags (&F, 0, 0.1, 0, 1e-5, 1000,
                            w, &result, &error); 

    return result;
    }

// Evaluate Function over Grid
int calculateGrid(double* d_list){
    omp_set_num_threads(THREAD_NUM); // set number of threads in "parallel" blocks
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                //d_list[j*nx + i] = f(0.2, cFI(i, dx, shift_x), cFI(j, dy, shift_y), vs1);
                //d_list[j*nx + i] = integrate_f(cFI(i, dx, shift_x), cFI(j, dy, shift_y), vs1);
                d_list[j*nx + i] = integrate_f_cond(cFI(i, dx, shift_x), cFI(j, dy, shift_y));
                //d_list[j*nx + i] = condensedLDOS(0.2, cFI(i, dx, shift_x), cFI(j, dy, shift_y), K0, 0);
            }
        }
    }

    return 0;
}

// Writeout to file
int save2file(double* d_list, string fname){
    string fodir = "output/sep" + std::to_string(int(V0)) + "a/";
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

int bFromSep(int sep){

    if (sep % 2 !=0){
        return int((sep+1)/2);
    }
    else{
        return 0;
    }
}

int main()
{
    //for (int i = 7; i < 8; i++){
    //    int bs = bFromSep(2*i + 1);
    //    //updateSep(bs);
    //    readRun();
    //}

    //updateSep(3);
    //calculateGrid(output);
    //save2file(output, "condensed");

    for (int i = 10; i < 11; i++){
        int bs = bFromSep(2*i + 1);
        //updateSep(i);
        cout << "Calculating for V0" << i;
        V0 = 10 *i;
        calculateGrid(output);
        save2file(output, "condensed");
    }

     
    return 0;
}