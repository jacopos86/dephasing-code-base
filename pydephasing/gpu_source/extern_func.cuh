#include <math.h>

/* bose occupation internal function */

__device__ double bose_occup(double x, double T, const double TOLER) {
    double nql;
    if (T < TOLER) {
        nql = 0.;
    }
    else {
        if (x > 100.) {
            nql = 0.;
        }
        else {
            nql = 1./(exp(x) - 1.);
        }
    }
    return nql;
}

/* lorentzian function */

__device__ double lorentzian(double x, double eta) {
    double ltz;
    ltz = 1./PI * eta / 2. / (x * x + eta * eta / 4.);
    return ltz;
}

