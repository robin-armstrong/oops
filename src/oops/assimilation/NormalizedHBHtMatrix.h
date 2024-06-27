/*
 * (C) Copyright 2009-2016 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef OOPS_ASSIMILATION_NORMALIZEDHBHTMATRIX_H_
#define OOPS_ASSIMILATION_NORMALIZEDHBHTMATRIX_H_

#include <memory>
#include <utility>

#include <boost/noncopyable.hpp>

#include "oops/assimilation/ControlIncrement.h"
#include "oops/assimilation/CostFunction.h"
#include "oops/assimilation/DualVector.h"
#include "oops/assimilation/RinvSqrtMatrix.h"
#include "oops/base/PostProcessorTLAD.h"
#include "oops/util/PrintAdjTest.h"

namespace oops {

/// The \f$ R^{-1/2} H B H^T R^{-1/2} \f$ matrix.
/*!
 *  The solvers represent matrices as objects that implement a "multiply"
 *  method. This class defines objects that apply a generalized
 *  \f$ R^{-1/2} H B H R^{-1/2} ^T\f$ matrix which includes \f$ H \f$ and
 *  the equivalent operators for the other terms of the cost function.
 */

template<typename MODEL, typename OBS> class NormalizedHBHtMatrix : private boost::noncopyable {
  typedef ControlIncrement<MODEL, OBS>    CtrlInc_;
  typedef CostFunction<MODEL, OBS>        CostFct_;
  typedef DualVector<MODEL, OBS>          Dual_;
  typedef RinvSqrtMatrix<MODEL, OBS>      R_invsqrt_;

 public:
  explicit NormalizedHBHtMatrix(const CostFct_ & j, const bool test = false);

  void multiply(const Dual_ & dy, Dual_ & dz) const;

 private:
  CostFct_ const & j_;
  bool test_;
  mutable int iter_;
  R_invsqrt_ RinvSqrt_;
};

// -----------------------------------------------------------------------------

template<typename MODEL, typename OBS>
NormalizedHBHtMatrix<MODEL, OBS>::NormalizedHBHtMatrix(const CostFct_ & j, const bool test)
  : j_(j), test_(test), iter_(0), RinvSqrt_(j)
{}

// -----------------------------------------------------------------------------

template<typename MODEL, typename OBS>
void NormalizedHBHtMatrix<MODEL, OBS>::multiply(const Dual_ & dy, Dual_ & dz) const {
// Increment counter
  iter_++;

// Setting up scratch space
  Dual_ dy_tmp(dy);

// Pre-multiply by R^{-1/2}
  RinvSqrt_.multiply(dy, dy_tmp);

// Run ADJ
  CtrlInc_ ww(j_.jb());
  j_.zeroAD(ww);
  PostProcessorTLAD<MODEL> costad;
  for (unsigned jj = 0; jj < j_.nterms(); ++jj) {
    j_.jterm(jj).computeCostAD(dy_tmp.getv(jj), ww, costad);
  }
  j_.runADJ(ww, costad);
  for (unsigned jj = 0; jj < j_.nterms(); ++jj) {
    j_.jterm(jj).setPostProcAD();
  }

// Multiply by B
  CtrlInc_ zz(j_.jb());
  j_.jb().multiplyB(ww, zz);

// Run TLM
  PostProcessorTLAD<MODEL> costtl;
  for (unsigned jj = 0; jj < j_.nterms(); ++jj) {
    j_.jterm(jj).setPostProcTL(zz, costtl);
  }

  CtrlInc_ mzz(zz);
  j_.runTLM(mzz, costtl);

// Get TLM outputs
  dy_tmp.clear();
  for (unsigned jj = 0; jj < j_.nterms(); ++jj) {
    std::unique_ptr<GeneralizedDepartures> ztmp = j_.jterm(jj).newDualVector();
    j_.jterm(jj).computeCostTL(zz, *ztmp);
    dy_tmp.append(std::move(ztmp));
  }

// Tests
  if (test_) {
    // <G dx, dy >, where dx = B Gt dy
    double adj_tst_fwd = dot_product(dy_tmp, dy);
    // <  dx, Gt dy>, where dx = B Gt dy
    double adj_tst_bwd = dot_product(zz, ww);

    Log::info() << "Online adjoint test, iteration: " << iter_ << std::endl
                << util::PrintAdjTest(adj_tst_fwd, adj_tst_bwd, "G") << std::endl;
  }

// Post-multiply by R^{-1/2}
  RinvSqrt_.multiply(dy_tmp, dz);
}

// -----------------------------------------------------------------------------

}  // namespace oops

#endif  // OOPS_ASSIMILATION_NORMALIZEDHBHTMATRIX_H_
