// anLDOS.cpp -- Numerical Evaluation of expression for modulations in local density of states for scattering off of two point defects in graphene
// Author: Anderson Steckler
// Date: June 23, 2025

#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

// Constants and Parameters
const float V0 = 1; // eV
const double VF = 9.060911856897319e14; // nm/s 
const float a = 0.24595; // nm
const int sep = 7; // nm separation constant
const float hop = -2.8; // eV
const double Hbar = 6.582119569e-16; // eV * s


gsl_complex Hankel0(double x){
  return gsl_complex_rect(gsl_sf_bessel_J0(x), gsl_sf_bessel_Y0(x));
}

int
main (void)
{

  gsl_complex result = Hankel0(0.23);

  // Create Parameters
  printf("%.4f + %.4f i\n", GSL_REAL(result), GSL_IMAG(result));
}

