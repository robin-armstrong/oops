/*
 * (C) Copyright 2009-2016 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef OOPS_ASSIMILATION_GAUSSLEGENDRE_H_
#define OOPS_ASSIMILATION_GAUSSLEGENDRE_H_

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <vector>

#include "oops/util/Logger.h"

namespace oops {

//-------------------------------------------------------------------------------------------------
// Computes nodes and weights for Gauss-Legendre quadrature using the Golub-Welsch algorithm.
void GaussLegendre(const int quadsize, std::vector<double>& nodes, std::vector<double>& weights) {
  Log::info() << "GaussLegendre: Starting and forming Golub-Welsch matrix." << std::endl;
  
  Eigen::MatrixXf GW(quadsize, quadsize);
  GW.setZero();

  for(int q = 0; q + 1 < quadsize; q++) {
    GW(q+1, q) = .5/sqrt(1 - pow(2*(q + 1), -2));
    GW(q, q+1) = GW(q+1, q);
  }

  Log::info() << "GaussLegendre: Eigenvalue decomposition of Golub-Welsch matrix." << std::endl;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(quadsize);
  solver.compute(GW);
  Eigen::VectorXf evals = solver.eigenvalues();
  Eigen::MatrixXf evecs = solver.eigenvectors();

  Log::info() << "GaussLegendre: Calculating quadrature weights and nodes." << std::endl;

  for(int q = 0; q < quadsize; q++) {
    nodes[q]   = evals(q);
    weights[q] = 2*pow(evecs(0, q), 2);
  }
}

}  // namespace oops

#endif  // OOPS_ASSIMILATION_GAUSSLEGENDRE_H_
