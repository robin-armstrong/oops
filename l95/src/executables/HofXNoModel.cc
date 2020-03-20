/*
 * (C) Copyright 2020 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "lorenz95/L95Traits.h"
#include "oops/runs/HofXNoModel.h"
#include "oops/runs/Run.h"

int main(int argc,  char ** argv) {
  oops::Run run(argc, argv);
  oops::HofXNoModel<lorenz95::L95Traits> hofx;
  run.execute(hofx);
  return 0;
}
