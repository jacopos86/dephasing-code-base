#include <math.h>
#define PI 3.141592653589793

/* bose occupation internal function */

__device__ double bose_occup(double x) {
    double nql;
    if (x > 100.) {
        nql = 0.;
    }
    else {
        nql = 1./(exp(x) - 1.);
    }
    return nql;
}

/* lorentzian function */

__device__ double lorentzian(double x, double eta) {
    double ltz = 1./PI * eta / 2. / (pow(x, 2) + pow(eta, 2) / 4.);
    return ltz;
}

/* gaussian function */

__device__ double gaussian(double x, double eta) {
    double g = exp(-pow(x, 2)/(2.0 * pow(eta, 2))) / sqrt(2.0 * PI * pow(eta, 2));
    return g;
}