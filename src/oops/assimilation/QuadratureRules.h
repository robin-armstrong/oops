/*
 * (C) Copyright 2024 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef OOPS_ASSIMILATION_QUADRATURERULES_H_
#define OOPS_ASSIMILATION_QUADRATURERULES_H_

#include <iostream>
#include <vector>

#include "oops/assimilation/EllipticFunctions.h"
#include "oops/util/Logger.h"

namespace oops {

/* Computes shifts and weights for the quadrature-based perturbation update.
 * Uses formulas from Hale, Higham, and Trefethen, "Computing $A^\alpha$,
 * $\log(A)$, and Related Matrix Functions by Contour Integrals," SIAM
 * Journal on Numerical Analysis, 2008. */

void prepare_quad_rule(std::vector<double>& shifts, std::vector<double>& weights,
                       const int quadsize, const double scale) {
  const double PI = 4*atan(1);
  
  shifts.clear();
  weights.clear();

  double r, s, u;
  double k  = 1/sqrt(1 + scale);
  double el = ellipk(sqrt(1 - k*k));

  for(int q = 0; q < quadsize; q++) {
    u = (q + 1 - .5)/quadsize;
    
    std::vector<double> sn_cn_dn = ellipj_imag(el*u, k);

    s = sn_cn_dn[0];
    s = s*s;
    r = 2*el*sn_cn_dn[1]*sn_cn_dn[2]/(PI*quadsize);

    shifts.push_back(s + 1);
    weights.push_back(r/(s + 1));
  }
}

}  // namespace oops

#endif  // OOPS_ASSIMILATION_QUADRATURERULES_H_
