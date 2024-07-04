/*
 * (C) Copyright 2024 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef OOPS_ASSIMILATION_QUADRATURESOLVER_H_
#define OOPS_ASSIMILATION_QUADRATURESOLVER_H_

#include <cfloat>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "oops/assimilation/ControlIncrement.h"
#include "oops/assimilation/DualVector.h"
#include "oops/assimilation/CostFunction.h"
#include "oops/assimilation/QuadratureRules.h"
#include "oops/assimilation/HMatrix.h"
#include "oops/assimilation/HtMatrix.h"
#include "oops/assimilation/BMatrix.h"
#include "oops/assimilation/RinvSqrtMatrix.h"
#include "oops/assimilation/NormalizedHBHtMatrix.h"
#include "oops/assimilation/SLCG.h"
#include "oops/util/Logger.h"

namespace oops {

template <typename MODEL, typename OBS>
class QuadratureSolver {
  typedef CostFunction<MODEL, OBS>          CostFct_;
  typedef ControlIncrement<MODEL, OBS>      CtrlInc_;
  typedef DualVector<MODEL, OBS>            Dual_;
  typedef HMatrix<MODEL, OBS>               H_;
  typedef HtMatrix<MODEL, OBS>              Ht_;
  typedef BMatrix<MODEL, OBS>               B_;
  typedef RinvSqrtMatrix<MODEL, OBS>        R_invsqrt_;
  typedef NormalizedHBHtMatrix<MODEL, OBS>  NormalHBHt_;

  public:
    QuadratureSolver(const CostFct_ & J);
    ~QuadratureSolver() {}
    ControlIncrement<MODEL, OBS> solve(CtrlInc_ dx, const int quadsize, const int maxiter, const double tol);

  private:
    H_          H_mat;
    Ht_         Ht_mat;
    B_          B_mat;
    R_invsqrt_  R_invsqrt_mat;
    NormalHBHt_ NormalHBHt_mat;
    double      pi;
};

// =============================================================================

template<typename MODEL, typename OBS>
QuadratureSolver<MODEL, OBS>::QuadratureSolver(const CostFct_ & J)
  : H_mat(J), Ht_mat(J), B_mat(J), R_invsqrt_mat(J), NormalHBHt_mat(J)
{
  this->pi = 4*atan(1);
}

// -----------------------------------------------------------------------------

template<typename MODEL, typename OBS>
ControlIncrement<MODEL, OBS> QuadratureSolver<MODEL, OBS>::solve(CtrlInc_ dx, const int quadsize, const int maxiter, const double tol) {
  Log::info() << "QuadratureSolver: Starting." << std::endl;
  Log::info() << "QuadratureSolver: Mapping increment to observation space." << std::endl;
  
  Dual_ dy, dz_in;
  H_mat.multiply(dx, dy);
  R_invsqrt_mat.multiply(dy, dz_in);

  Log::info() << "QuadratureSolver: Setting up quadrature rule." << std::endl;
// Set up nodes and weights for Gauss-Legendre quadrature
  std::vector<double> weights;
  std::vector<double> nodes;
  gaussLegendre(quadsize, nodes, weights);
  prepareEAKFQuad(quadsize, nodes, weights);

  Log::info() << "QuadratureSolver: Beginning linear system solves." << std::endl;
  std::vector<Dual_> dz_out;
  SLCG(dz_out, NormalHBHt_mat, dz_in, nodes, quadsize, maxiter, tol);

  Log::info() << "QuadratureSolver: Recombining solutions." << std::endl;

  Dual_ final_inc_obspace(dz_out[0]);
  final_inc_obspace.zero();

  for(int q = 0; q < quadsize; q++) {
    dz_out[q]         *= weights[q];
    final_inc_obspace += dz_out[q];
  }

  Log::info() << "QuadratureSolver: Mapping increment to state space." << std::endl;

  Dual_    tmp_dual(final_inc_obspace);
  CtrlInc_ tmp_ctrl(dx), dx_out(dx);

  R_invsqrt_mat.multiply(final_inc_obspace, tmp_dual);
  Ht_mat.multiply(tmp_dual, tmp_ctrl);
  B_mat.multiply(tmp_ctrl, dx_out);

  Log::info() << "QuadratureSolver: Applying adjustment to input increment." << std::endl;

  dx_out *= -1;
  dx_out += dx;

  Log::info() << "QuadratureSolver: finished." << std::endl;
  return dx_out;
}

}  // namespace oops
#endif  // OOPS_ASSIMILATION_QUADRATURESOLVER_H_
