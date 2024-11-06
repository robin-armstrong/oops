/*
 * (C) Copyright 2024 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "lorenz95/instantiateLocalizationFactory.h"
#include "lorenz95/L95Traits.h"
#include "oops/runs/Run.h"
#include "oops/runs/QuadratureUpdate.h"

int main(int argc,  char ** argv) {
  oops::Run run(argc, argv);
  lorenz95::instantiateLocalizationFactory();
  oops::QuadratureUpdate<lorenz95::L95Traits, lorenz95::L95ObsTraits> var;
  return run.execute(var);
}
