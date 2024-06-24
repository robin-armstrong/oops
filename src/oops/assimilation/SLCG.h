/*
 * (C) Copyright 2009-2016 ECMWF.
 * (C) Crown Copyright 2024, the Met Office.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0. 
 * In applying this licence, ECMWF does not waive the privileges and immunities 
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef OOPS_ASSIMILATION_SLCG_H_
#define OOPS_ASSIMILATION_SLCG_H_

#include <cmath>
#include <iostream>
#include <vector>

#include "oops/assimilation/MinimizerUtils.h"
#include "oops/assimilation/rotmat.h"
#include "oops/assimilation/UpTriSolve.h"
#include "oops/util/dot_product.h"
#include "oops/util/Logger.h"

namespace oops {

/*! \file SLCG.h
 * \brief SLCG solver for (A + lambda*I)x=b, for multiple values of lambda.
 *
 * Solves (A + lambda*I)x=b for several values of lambda in parallel, using
 * a shifted variant of the Lanczos conjugate gradients algorithm (Golub &
 * Van Loan, "Matrix Computations" (1996), section 9.3.1).

 * A must be square and symmetric positive-definite, and each lambda must 
 * be nonnegative. Alternately, A can be symmetric positive-semidefinite,
 * in which case each lambda must be strictly positive.
 *
 * On entry:
 * -    X        = vector of solutions for different shift values,
 *                 initialized externally and modified by this function.
 * -    A        = \f$ A \f$.
 * -    b        = right hand side.
 * -    lambda   = vector of shift values.
 * -    nlambdas = number of shift values.
 * -    maxiter  = maximum number of iterations
 * -    tol      = error tolerance
 *
 * On exit, X[j] will contain the solution to (A + lambda[j]I)x = b for
 * j = 0, 1, ..., (nlambdas - 1). The return value is the achieved reduction
 * in residual norm, measured with respect to the worst-conditioned system 
 * being solved, i.e., the system with the smallest value of lambda.
 *
 * Iteration will stop if the maximum iteration limit "maxiter" is reached
 * or if the residual norm reduces by a factor of "tolerance".
 *
 * VECTOR must implement:
 *  - dot_product
 *  - operator(=)
 *  - operator(+=),
 *  - operator(-=)
 *  - operator(*=) [double * VECTOR],
 *  - axpy
 *
 *  AMATRIX must implement a method:
 *  - void multiply(const VECTOR&, VECTOR&) const
 *
 *  which applies the matrix to the first argument, and returns the
 *  matrix-vector product in the second. (Note: the const is optional, but
 *  recommended.)
 */

template <typename VECTOR, typename AMATRIX>
double SLCG(std::vector<VECTOR> & X, const AMATRIX & A, 
            const VECTOR & b, const std::vector<double> lambda,
            const int nlambdas, const int maxiter,
            const double tolerance) {  
  double              alpha;
  double              beta;
  std::vector<double> d;
  std::vector<double> l;
  std::vector<double> v;
  VECTOR              q(b);
  VECTOR              rho(b);
  VECTOR              u(b);
  VECTOR              prod(b);
  VECTOR              testvec(b);
  VECTOR              r(b);
  std::vector<VECTOR> C;

  X.clear();

  /* The first block of code initializes all the solver
   * data and, simultaneously, performs the first iteration. */

  Log::info() << " SLCG Starting Iteration 1 and initializing" << std::endl;

  double bnrm2 = dot_product(b, b);

  // initializing data used by the Lanczos process
  beta  = sqrt(bnrm2);
  u    *= 0;
  q     = b; q *= (1/beta);

  A.multiply(q, prod);
  alpha = dot_product(q, prod);

  rho = prod;
  rho.axpy(-alpha, q);
  rho.axpy(-beta, u);

  // setting up the next vector that we'll need a matrix-vector product with
  testvec = rho; testvec *= (1/sqrt(dot_product(rho, rho)));

  double min_lambda    = lambda[0];
  int min_lambda_index = 0;

  /* initializing implicit LDLt factorization of A's Krylov projection,
   * along with solutions and search directions. */
  for(int i = 0; i < nlambdas; i++) {
    if(lambda[i] < min_lambda) {
      min_lambda       = lambda[i];
      min_lambda_index = i;
    }

    d.push_back(alpha + lambda[i]);
    v.push_back(beta/d[i]);
    l.push_back(0.);
    C.push_back(q);
    X.push_back(C[i]);
    X[i] *= v[i];
  }

  // residual vector for worst conditioned system being solved
  r = rho; r *= v[min_lambda_index];

  double rnrm2         = dot_product(rho, rho);
  double normReduction = rnrm2 / bnrm2;

  printNormReduction(1, rnrm2, normReduction);
  
  Log::info() << std::endl;

  if (normReduction <= tolerance) {
    Log::info() << "SLCG: Achieved required reduction in residual norm." << std::endl;
  } else {
    // SLCG iteration
    for (int jiter = 1; jiter < maxiter; ++jiter) {
      Log::info() << " SLCG Starting Iteration " << jiter+1 << std::endl;

      // updating the Lanczos process
      A.multiply(testvec, prod);
      u     = q;
      q     = testvec;
      alpha = dot_product(q, prod);
      beta  = sqrt(dot_product(rho, rho));
      rho   = prod; rho.axpy(-alpha, q); rho.axpy(-beta, u);
      
      // setting up the next vector that we'll need a matrix-vector product with
      testvec  = rho;
      testvec *= 1/sqrt(dot_product(rho, rho));

      // updating implicit LDLt factorization of A's Krylov projection
      for(int i = 0; i < nlambdas; i++) {
        l[i] = beta/d[i];
      }

      for(int i = 0; i < nlambdas; i++) {
        d[i] = alpha + lambda[i] - beta*l[i];
      }

      // updating search directions and solutions
      for(int i = 0; i < nlambdas; i++) {
        C[i] *= -l[i];
        C[i] += q;
        v[i] *= -beta/d[i];

        X[i].axpy(v[i], C[i]);
      }

      // updating residual of worst conditioned system
      r     = rho; r *= v[min_lambda_index];
      rnrm2 = dot_product(r, r);

      // checking termination conditions
      normReduction = rnrm2/bnrm2;
      Log::info() << "SLCG end of iteration " << jiter+1 << std::endl;
      printNormReduction(jiter+1, rnrm2, normReduction);

      if (normReduction <= tolerance) {
          Log::info() << "SLCG: Achieved required reduction in residual norm." << std::endl;
          jiter += 1;
          break;
      }
    }
  }

  Log::info() << "SLCG: end" << std::endl;

  return normReduction;
}

}  // namespace oops

#endif  // OOPS_ASSIMILATION_SLCG_H_
