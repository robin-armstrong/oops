/*
 * (C) Copyright 2024 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef OOPS_ASSIMILATION_ELLIPTICFUNCTIONS_H_
#define OOPS_ASSIMILATION_ELLIPTICFUNCTIONS_H_

#include <iostream>
#include <vector>

#include "oops/util/Logger.h"

namespace oops {

/* Evaluates Carlson's elliptic integral of the first kind. Uses the
 * algorithm from Press et al., "Numerical Recipes in Fortran," second
 * edition, section 6.1.1. */

double ellip_rf(double x0, double y0, double z0) {
    const int    MAXITER = 100;
    const double TOL     = 1e-16;
    
    double x   = x0,
           y   = y0,
           z   = z0;
    double lam = sqrt(x*y) + sqrt(x*z) + sqrt(y*z);

    double mu  = (x + y + z)/3;
    double ex  = 1 - x/mu,
           ey  = 1 - y/mu,
           ez  = 1 - z/mu;
    double eps = std::max(ex*ex, std::max(ey*ey, ez*ez));

    int iter = 1;

    while ((iter < MAXITER) && (eps > TOL)) {
        x   = .25*(x + lam);
        y   = .25*(y + lam);
        z   = .25*(z + lam);
        lam = sqrt(x*y) + sqrt(x*z) + sqrt(y*z);

        mu  = (x + y + z)/3;
        ex  = 1 - x/mu;
        ey  = 1 - y/mu;
        ez  = 1 - z/mu;
        eps = std::max(ex*ex, std::max(ey*ey, ez*ez));

        iter++;
    }

    if (iter == MAXITER) {
        Log::info() << std::endl
                    << "EllipticFunctions: elliptic_rf reached maximum iteration count"
                    << std::endl;
    }

    double E2 = ex*ey - ez*ez;
    double E3 = ex*ey*ez;
    double Rf = (1 - E2/10 + E3/14 + E2*E2/24 - 3*E2*E3/44)/sqrt(mu);

    return Rf;
}

// Evaluates the complete elliptic integral of the first kind.

double ellipk(double k) {
    if (abs(k) >= 1) {
        Log::info() << std::endl
                    << "EllipticFunctions: modulus for ellip_k is not in (0, 1), returning NaN"
                    << std::endl;
        return NAN;
    }

    return ellip_rf(0, 1 - k*k, 1);
}

/* Evaluates the Jacobian elliptic functions on a real argument.
 * Uses the algorithm from "Transformation of the Jacobian Amplitude
 * Function And Its Calculation via the Arithmetic-Geometric Mean,"
 * Kenneth J. Sala, SIAM Journal on Mathematical Analysis (1989). */

std::vector<double> ellipj_real(double u, double k) {
    if (abs(k) >= 1) {
        Log::info() << std::endl
                    << "EllipticFunctions: modulus for ellipj_real is not in (0, 1), returning NaN"
                    << std::endl;
        
        std::vector<double> nan_arr(3, NAN);
        return nan_arr;
    }
    
    const int MAXITER = 100;
    const double TOL  = 1e-16;

    std::vector<double> a(MAXITER+1, 0.);
    std::vector<double> b(MAXITER+1, 0.);
    std::vector<double> c(MAXITER+1, 0.);
    std::vector<double> psi(MAXITER+1, 0.);

    a[0]   = 1;
    b[0]   = sqrt(1 - k*k);
    c[0]   = k;
    psi[0] = u;

    int i = 0;

    while ((i < MAXITER) && (abs(c[i]) > TOL)) {
        a[i+1]   = .5*(a[i] + b[i]);
        b[i+1]   = sqrt(a[i]*b[i]);
        c[i+1]   = .5*(a[i] - b[i]);
        psi[i+1] = 2*psi[i];
        
        i++;
    }

    if (i == MAXITER) {
        Log::info() << std::endl
                    << "EllipticFunctions: ellipj_real reached maximum iteration count"
                    << std::endl;
    }

    psi[i] *= a[i];

    while (i > 0) {
        psi[i-1] = .5*(psi[i] + asin(c[i]*sin(psi[i])/a[i]));
        i--;
    }

    double sn = sin(psi[0]);
    double cn = cos(psi[0]);
    double dn = cn/cos(psi[0] - psi[1]);

    std::vector<double> sn_cn_dn(3, 0.);
    
    sn_cn_dn[0] = sn;
    sn_cn_dn[1] = cn;
    sn_cn_dn[2] = dn;

    return sn_cn_dn;
}

/* Evaluates the Jacobian elliptic functions on an imaginary argument
 * using Jacobi's imaginary transformation. Returns the imaginary part
 * of sn, and the real parts of cn and dn. */

std::vector<double> ellipj_imag(double u, double k) {
    double kp = sqrt(1 - k*k);
    std::vector<double> ellipj_kp = ellipj_real(u, kp);
    std::vector<double> sn_cn_dn(3, 0.);

    sn_cn_dn[0] = ellipj_kp[0]/ellipj_kp[1];
    sn_cn_dn[1] = 1/ellipj_kp[1];
    sn_cn_dn[2] = ellipj_kp[2]/ellipj_kp[1];

    return sn_cn_dn;
}

}  // namespace oops

#endif  // OOPS_ASSIMILATION_ELLIPTICFUNCTIONS_H_
