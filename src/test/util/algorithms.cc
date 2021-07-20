/*
 * (C) Crown copyright 2021, Met Office
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "oops/runs/Run.h"
#include "test/util/algorithms.h"

int main(int argc, char **argv) {
  oops::Run run(argc, argv);
  test::Algorithms tests;
  return run.execute(tests);
}