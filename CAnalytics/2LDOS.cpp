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

//////////////////////
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

////////////////////////
// Physics Parameters //
///////////////////////
const double V0 = 1; // eV
const double VF = 9.060911856897319e14; // nm/s 
const double a = 0.24595; // nm
const double acc = 0.142; //nm
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

// Change the separation and all the variables related to separation
int updateSep(int sepnum){

    // Separation is an integer related to number of unit cells in between the defect's location
    // Two unit cells are seperated by the width of the graphene hexagon a_cc * tan(60)
    double h = acc * tan(M_PI/3);

    sep = sepnum; // nm separation constant
    R1x = -sep * h;
    R1y = 0;
    R2x = sep *h;
    R2y = 0;

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
gsl_complex G0AA(double w){
    double real_part = (M_PI * w)/(pow(VF*Hbar, 2)) * log(pow(w, 2)/(pow(hop, 2) - pow(w, 2)));
    double imag_part = -(pow(M_PI, 2)*abs(w))/(pow(VF*Hbar, 2));

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
gsl_complex Rfrac(double w, double K1x, double K1y, double K2x, double K2y){

    double d12x = R1x - R2x;
    double d12y = R1y - R2y;

    GFParams p1 = {K1x, K1y, d12x, d12y, "AA"};
    GFParams p2 = {K2x, K2y, -d12x, -d12y, "AA"};
    GFParams p3 = {K1x, K1y, d12x, d12y, "AB"};
    GFParams p4 = {K2x, K2y, -d12x, -d12y, "BA"};

    gsl_complex tnum = telem(0.2);
    gsl_complex tnum2 = gsl_complex_mul(tnum, tnum);

    gsl_complex add4 = gsl_complex_mul(tnum2, gsl_complex_add(orderone(w, p1, p2), orderone(w, p3, p4)));
    gsl_complex denom = gsl_complex_add(gsl_complex_rect(1, 0), gsl_complex_mul(gsl_complex_rect(-1, 0), add4));

    return gsl_complex_div(gsl_complex_rect(1, 0), denom);
}

// Caluclating and summing all the LDOS terms
double f(double w, double x, double y, vScheme vs){

    double dR1x = x-R1x;
    double dR1y = y-R1y;
    double dR2x = x - R2x;
    double dR2y = y - R2y;
    double d12x = R1x - R2x;
    double d12y = R1y - R2y;

    ///////////////////////////
    //// First Order Terms ////
    ///////////////////////////

    ///////////////////////
    // alpha tilde alpha //
    ///////////////////////
    GFParams p1 = {vs.Katx, vs.Katy, -dR1x, -dR1y, "AA"};
    GFParams p2 = {vs.Kax, vs.Kay, dR1x, dR1y, "AA"};
    gsl_complex term1 = orderone(w, p1, p2);

    p1 = {vs.Katx, vs.Katy, -dR1x, -dR1y, "AB"};
    p2 = {vs.Kax, vs.Kay, dR1x, dR1y, "BA"};
    gsl_complex term2 = orderone(w, p1, p2);

    /////////////////////
    // beta tilde beta //
    /////////////////////
    p1 = {vs.Kbtx, vs.Kbty, -dR2x, -dR2y, "AA"};
    p2 = {vs.Kbx, vs.Kby, dR2x, dR2y, "AA"};
    gsl_complex term3 = orderone(w, p1, p2);

    p1 = {vs.Kbtx, vs.Kbty, -dR2x, -dR2y, "AB"};
    p2 = {vs.Kbx, vs.Kby, dR2x, dR2y, "BA"};
    gsl_complex term4 = orderone(w, p1, p2);

    gsl_complex pfo1 = gsl_complex_mul(Rfrac(w, vs.Kgx, vs.Kgy, vs.Kgtx, vs.Kgty), telem(w));
    gsl_complex term_order_one = gsl_complex_add(gsl_complex_add(gsl_complex_add(term4, term3), term2), term1);
    gsl_complex cp1 = gsl_complex_mul(pfo1, term_order_one);

    ////////////////////////////
    //// Second Order Terms ////
    ////////////////////////////

    /////////////
    // gt at b //
    /////////////
    p1 = {vs.Kgtx, vs.Kgty, -d12x, -d12y, "AA"};
    p2 = {vs.Katx, vs.Katy, -dR1x, -dR1y, "AA"};
    GFParams p3 = {vs.Kbx, vs.Kby, dR2x, dR2y, "AA"};
    gsl_complex term5 = ordertwo(w, p1, p2, p3);

    p1 = {vs.Kgtx, vs.Kgty, -d12x, -d12y, "AA"};
    p2 = {vs.Katx, vs.Katy, -dR1x, -dR1y, "AB"};
    p3 = {vs.Kbx, vs.Kby, dR2x, dR2y, "BA"};
    gsl_complex term6 = ordertwo(w, p1, p2, p3);
    
    /////////////
    // g bt a //
    /////////////
    p1 = {vs.Kgx, vs.Kgy, d12x, d12y, "AA"};
    p2 = {vs.Kbtx, vs.Kbty, -dR2x, -dR2y, "AA"};
    p3 = {vs.Kax, vs.Kay, dR1x, dR1y, "AA"};
    gsl_complex term7 = ordertwo(w, p1, p2, p3);

    p1 = {vs.Kgx, vs.Kgy, d12x, d12y, "AA"};
    p2 = {vs.Kbtx, vs.Kbty, -dR2x, -dR2y, "AB"};
    p3 = {vs.Kax, vs.Kay, dR1x, dR1y, "BA"};
    gsl_complex term8 = ordertwo(w, p1, p2, p3);

    gsl_complex pfo2 = gsl_complex_mul(pfo1, telem(w));
    gsl_complex term_order_two = gsl_complex_add(gsl_complex_add(gsl_complex_add(term8, term7), term6), term5);
    gsl_complex cp2 = gsl_complex_mul(pfo2, term_order_two);

    /////////////////////////////
    // Appended 2nd Order Terms//
    /////////////////////////////
    /////////////
    // gt at b //
    /////////////
    p1 = {vs.Kgtx, vs.Kgty, -d12x, -d12y, "AB"};
    p2 = {vs.Katx, vs.Katy, -dR1x, -dR1y, "BA"};
    p3 = {vs.Kbx, vs.Kby, dR2x, dR2y, "AA"};
    gsl_complex term9 = ordertwo(w, p1, p2, p3);


    p1 = {vs.Kgtx, vs.Kgty, -d12x, -d12y, "AB"};
    p2 = {vs.Katx, vs.Katy, -dR1x, -dR1y, "BB"};
    p3 = {vs.Kbx, vs.Kby, dR2x, dR2y, "BA"};
    gsl_complex term10 = ordertwo(w, p1, p2, p3);

    /////////////
    // g bt a //
    /////////////
    p1 = {vs.Kgx, vs.Kgy, d12x, d12y, "AB"};
    p2 = {vs.Kbtx, vs.Kbty, -dR2x, -dR2y, "BA"};
    p3 = {vs.Kax, vs.Kay, dR1x, dR1y, "AA"};
    gsl_complex term11 = ordertwo(w, p1, p2, p3);


    p1 = {vs.Kgx, vs.Kgy, d12x, d12y, "AB"};
    p2 = {vs.Kbtx, vs.Kbty, -dR2x, -dR2y, "BB"};
    p3 = {vs.Kax, vs.Kay, dR1x, dR1y, "BA"};
    gsl_complex term12 = ordertwo(w, p1, p2, p3);

    gsl_complex term_order_two_extra = gsl_complex_add(gsl_complex_add(gsl_complex_add(term12, term11), term10), term9);
    gsl_complex cp3 = gsl_complex_mul(pfo2, term_order_two_extra);


    return -1*GSL_IMAG(gsl_complex_add(gsl_complex_add(cp1, cp2), cp3));
}

// Evaluate Function over Grid
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

// Writeout to file
int save2file(double* d_list, string fname){
    string fodir = "output/sep" + std::to_string(sep) + "a/";
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

// We use the GF for just one K point, so we need to assign each
// propagator a valley
// Valley Scheme: a, at, b, bt, g, gt
int setVScheme(string valleys){
    std::istringstream iss(valleys);
    std::string token;

    int i = 0;
    while (iss >> token){
        double Kx = 0;
        double Ky = 0;
        if (token == "K"){
            Kx = K0;
            Ky = 0;
        }
        else if (token == "P"){
            Kx = -K0;
            Ky = 0;
        }
        switch(i){
            case 0: vs1.Kax = Kx; vs1.Kay = Ky; break;
            case 1: vs1.Katx = Kx; vs1.Katy = Ky; break;
            case 2: vs1.Kbx = Kx; vs1.Kby = Ky; break;
            case 3: vs1.Kbtx = Kx; vs1.Kbty = Ky; break;
            case 4: vs1.Kgx = Kx; vs1.Kgy = Ky; break;
            case 5: vs1.Kgtx = Kx; vs1.Kgty = Ky; break;
            default: std::cerr << "Too many tokens!"; break;
        }

        i++;
    }

    return 0;
}

// We thorugh the file of all possible valley configurations (64)
// and calculate the LDOS and then save at the end we sum them together.
int readRun(){

    std::cout << "Calculating for " << sep << "u\n";

    std::ifstream infile("combinations.txt");

    if(!infile){
        std::cerr << "Failed to open file.\n";
    }

    string line;
    while (std::getline(infile, line)){
        std::cout << "Calculating for " << line << "...\n";
        setVScheme(line);
        calculateGrid(output);
        save2file(output, line);
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
    for (int i = 10; i < 16; i++){
        int bs = bFromSep(2*i + 1);
        updateSep(bs);
        readRun();
    }
     
    return 0;
}