/*
 * (C) Copyright 2009-2016 ECMWF.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0. 
 * In applying this licence, ECMWF does not waive the privileges and immunities 
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include "lorenz95/ObsBiasCovariance.h"

#include <cmath>
#include <iostream>
#include <string>

#include "eckit/config/Configuration.h"
#include "eckit/exception/Exceptions.h"

#include "lorenz95/ObsBiasCorrection.h"
#include "oops/util/Logger.h"
#include "oops/util/Random.h"

// -----------------------------------------------------------------------------
namespace lorenz95 {
// -----------------------------------------------------------------------------
ObsBiasCovariance::ObsBiasCovariance(const ObsTableView &, const eckit::Configuration & conf)
  : variance_(0.0), active_(false)
{
  if (conf.has("standard_deviation")) {
    active_ = true;
    const double zz = conf.getDouble("standard_deviation");
    variance_ = zz * zz;
    ASSERT(variance_ > 0.0);
    oops::Log::info() << "ObsBiasCovariance variance = " << variance_ << std::endl;
  }
}
// -----------------------------------------------------------------------------
void ObsBiasCovariance::multiply(const ObsBiasCorrection & dxin,
                                 ObsBiasCorrection & dxout) const {
  if (active_) {
    dxout = dxin;
    dxout *= variance_;
  } else {
    dxout.zero();
  }
}
// -----------------------------------------------------------------------------
void ObsBiasCovariance::inverseMultiply(const ObsBiasCorrection & dxin,
                                        ObsBiasCorrection & dxout) const {
  if (active_) {
    dxout = dxin;
    dxout *= 1.0 / variance_;
  } else {
    dxout.zero();
  }
}
// -----------------------------------------------------------------------------
void ObsBiasCovariance::randomize(ObsBiasCorrection & dx) const {
  if (active_) {
    util::NormalDistribution<double> x(1, 0.0, 1.0, 4);
    dx.value() = x[0] * std::sqrt(variance_);
  } else {
    dx.zero();
  }
}
// -----------------------------------------------------------------------------
void ObsBiasCovariance::print(std::ostream & os) const {
  if (active_) {
    os << "ObsBiasCovariance: variance = " << variance_;
  } else {
    os << "ObsBiasCovariance not active";
  }
}
// -----------------------------------------------------------------------------
}  // namespace lorenz95
