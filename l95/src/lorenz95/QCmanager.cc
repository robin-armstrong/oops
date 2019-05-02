/*
 * (C) Copyright 2018-2019 UCAR
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0. 
 */

#include "lorenz95/QCmanager.h"

#include <string>

#include "lorenz95/L95Traits.h"
#include "oops/interface/ObsFilter.h"

// -----------------------------------------------------------------------------
namespace lorenz95 {
// -----------------------------------------------------------------------------
static oops::FilterMaker<L95Traits, oops::ObsFilter<L95Traits, QCmanager> > makerQCm_("QCmanager");
// -----------------------------------------------------------------------------
}  // namespace lorenz95

