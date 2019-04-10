/*
 * (C) Copyright 2009-2016 ECMWF.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0. 
 * In applying this licence, ECMWF does not waive the privileges and immunities 
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef OOPS_ASSIMILATION_COSTTERMBASE_H_
#define OOPS_ASSIMILATION_COSTTERMBASE_H_

#include <boost/shared_ptr.hpp>

#include "oops/assimilation/ControlIncrement.h"
#include "oops/assimilation/ControlVariable.h"
#include "oops/base/PostBase.h"
#include "oops/base/PostBaseTLAD.h"
#include "oops/interface/Geometry.h"
#include "oops/interface/Increment.h"
#include "oops/interface/State.h"

namespace eckit {
  class Configuration;
}

namespace oops {

// -----------------------------------------------------------------------------

/// Base Class for Cost Function Terms
/*!
 * Abstract base class for the terms of the cost function.
 */

template<typename MODEL> class CostTermBase {
  typedef Geometry<MODEL>            Geometry_;
  typedef State<MODEL>               State_;
  typedef Increment<MODEL>           Increment_;
  typedef boost::shared_ptr<PostBase<State_> >    PostPtr_;
  typedef boost::shared_ptr<PostBaseTLAD<MODEL> > PostPtrTLAD_;

 public:
/// Destructor
  virtual ~CostTermBase() {}

/// Initialize before nonlinear model integration.
  virtual PostPtr_ initialize(const ControlVariable<MODEL> &, const eckit::Configuration &) = 0;
  virtual PostPtrTLAD_ initializeTraj(const ControlVariable<MODEL> &,
                                      const Geometry_ &, const eckit::Configuration &) = 0;

/// Finalize computation after nonlinear model integration.
  virtual double finalize() = 0;
  virtual void finalizeTraj() = 0;

/// Initialize before starting the TL run.
  virtual PostPtrTLAD_ setupTL(const ControlIncrement<MODEL> &) const = 0;

/// Initialize before starting the AD run.
  virtual PostPtrTLAD_ setupAD(boost::shared_ptr<const GeneralizedDepartures>,
                               ControlIncrement<MODEL> &) const = 0;

/// Multiply by covariance (or weight) matrix and its inverse.
  virtual GeneralizedDepartures * multiplyCovar(const GeneralizedDepartures &) const = 0;
  virtual GeneralizedDepartures * multiplyCoInv(const GeneralizedDepartures &) const = 0;

/// Provide new dual space vector (for example a Departure for Jo).
  virtual GeneralizedDepartures * newDualVector() const = 0;

/// Gradient at first guess.
  virtual GeneralizedDepartures * newGradientFG() const = 0;

/// Reset trajectory.
  virtual void resetLinearization() = 0;
};

// -----------------------------------------------------------------------------

}  // namespace oops

#endif  // OOPS_ASSIMILATION_COSTTERMBASE_H_
