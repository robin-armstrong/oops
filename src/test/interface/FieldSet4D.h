/*
 * (C) Copyright 2023 UCAR-
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define ECKIT_TESTING_SELF_REGISTER_CASES 0

#include <boost/noncopyable.hpp>

#include "eckit/config/LocalConfiguration.h"
#include "eckit/testing/Test.h"
#include "oops/base/FieldSet3D.h"
#include "oops/base/FieldSet4D.h"
#include "oops/base/Geometry.h"
#include "oops/base/Increment.h"
#include "oops/base/Increment4D.h"
#include "oops/base/State.h"
#include "oops/base/State4D.h"
#include "oops/base/Variables.h"
#include "oops/mpi/mpi.h"
#include "oops/runs/Test.h"
#include "oops/util/DateTime.h"
#include "oops/util/FieldSetOperations.h"
#include "oops/util/Logger.h"
#include "test/TestEnvironment.h"

namespace test {

// -----------------------------------------------------------------------------
/// \brief Tests FieldSet3D ctor (from State and Increment fieldsets), valid time
/// and variables.
template <typename MODEL> void testFieldSet3D() {
  typedef oops::State<MODEL>     State_;
  typedef oops::Increment<MODEL> Increment_;
  typedef oops::Geometry<MODEL>  Geometry_;

  const eckit::Configuration & config = TestEnvironment::config();
  const bool parallel = config.getBool("parallel 4D");

  // Only run the 3D test in non-parallel mode.
  if (!parallel) {
    const Geometry_ geometry(config.getSubConfiguration("geometry"), oops::mpi::world());
    const State_ xx1(geometry, config.getSubConfiguration("state"));
    oops::Log::info() << "State: " << xx1 << std::endl;
    const oops::FieldSet3D xx2(xx1.fieldSet(), xx1.validTime(), geometry.getComm());
    oops::Log::info() << "FieldSet3D: " << xx2 << std::endl;

    // Check that the valid times are the same:
    EXPECT(xx2.validTime() == xx1.validTime());

    // Check that the variables are the same:
    oops::Log::info() << "State variables: " << xx1.variables() << std::endl;
    oops::Log::info() << "FieldSet3D variables: " << xx2.variables() << std::endl;
    // Currently only comparing the variable strings since FieldSet3D provides levels in addition.
    // TODO(Algo): change to comparing .variables() when all models support State/Increment
    // variables that provide levels information.
    EXPECT(xx2.variables().variables() == xx1.variables().variables());

    const oops::Variables incvars(config, "increment variables");
    const util::DateTime inctime(2020, 1, 1, 0, 0, 0);
    const Increment_ dx1(geometry, incvars, inctime);
    oops::Log::info() << "Increment: " << dx1 << std::endl;
    const oops::FieldSet3D dx2(dx1.fieldSet(), dx1.validTime(), geometry.getComm());
    oops::Log::info() << "FieldSet3D: " << dx2 << std::endl;

    // Check that the valid times are the same:
    EXPECT(dx2.validTime() == dx1.validTime());

    // Check that the variables are the same:
    oops::Log::info() << "Increment variables: " << dx1.variables() << std::endl;
    oops::Log::info() << "FieldSet3D variables: " << dx2.variables() << std::endl;
    // Currently only comparing the variable strings since FieldSet3D provides levels in addition.
    // TODO(Algo): change to comparing .variables() when all models support State/Increment
    // variables that provide levels information.
    EXPECT(dx2.variables().variables() == dx1.variables().variables());
  }
}

// -----------------------------------------------------------------------------
/// \brief Tests FieldSet4D ctor (from State4D and Increment4D), valid times and
/// variables.
template <typename MODEL> void testFieldSet4D() {
  typedef oops::State4D<MODEL>     State4D_;
  typedef oops::Increment4D<MODEL> Increment4D_;
  typedef oops::Geometry<MODEL>    Geometry_;

  const eckit::Configuration & config = TestEnvironment::config();
  const bool parallel = config.getBool("parallel 4D");

  // Define space and time communicators
  const eckit::mpi::Comm * commSpace = &oops::mpi::world();
  const eckit::mpi::Comm * commTime = &oops::mpi::myself();
  if (parallel) {
    size_t ntasks = oops::mpi::world().size();
    size_t nslots = config.getInt("number of time slots per task");
    ASSERT(ntasks % nslots == 0);
    size_t myrank = oops::mpi::world().rank();
    size_t ntaskpslot = ntasks / nslots;
    size_t myslot = myrank / ntaskpslot;

    // Create a communicator for same sub-window, to be used for communications in space
    std::string sgeom = "comm_geom_" + std::to_string(myslot);
    char const *geomName = sgeom.c_str();
    commSpace = &oops::mpi::world().split(myslot, geomName);
    ASSERT(commSpace->size() == ntaskpslot);

    // Create a communicator for same local area, to be used for communications in time
    size_t myarea = commSpace->rank();
    std::string stime = "comm_time_" + std::to_string(myarea);
    char const *timeName = stime.c_str();
    commTime = &oops::mpi::world().split(myarea, timeName);
    ASSERT(commTime->size() == nslots);
  }

  const Geometry_ geometry(config.getSubConfiguration("geometry"), *commSpace, *commTime);
  const State4D_ xx1(geometry, config.getSubConfiguration("state4d"), *commTime);
  oops::Log::info() << "State4D: " << xx1 << std::endl;
  oops::FieldSet4D xx2(xx1);
  oops::Log::info() << "FieldSet4D: " << xx2 << std::endl;

  // Check that the valid times are the same:
  EXPECT(xx2.validTimes() == xx1.validTimes());

  // Check that the variables are the same:
  oops::Log::info() << "State4D variables: " << xx1.variables() << std::endl;
  oops::Log::info() << "FieldSet4D variables: " << xx2.variables() << std::endl;
  // Currently only comparing the variable strings since FieldSet3D provides levels in addition.
  // TODO(Algo): change to comparing .variables() when all models support State/Increment
  // variables that provide levels information.
  EXPECT(xx2.variables().variables() == xx1.variables().variables());

  // For Increment4D test, use zero increments with variables specified in yaml,
  // and the same times as in State4D
  const oops::Variables incvars(config, "increment variables");
  const Increment4D_ dx1(geometry, incvars, xx1.times(), *commTime);
  oops::Log::info() << "Increment4D: " << dx1 << std::endl;
  oops::FieldSet4D dx2(dx1);
  oops::Log::info() << "FieldSet4D: " << dx2 << std::endl;

  // Check that the valid times are the same:
  EXPECT(dx2.validTimes() == dx1.validTimes());

  // Check that the variables are the same:
  oops::Log::info() << "Increment4D variables: " << dx1.variables() << std::endl;
  oops::Log::info() << "FieldSet4D variables: " << dx2.variables() << std::endl;
  // Currently only comparing the variable strings since FieldSet3D provides levels in addition.
  // TODO(Algo): change to comparing .variables() when all models support State/Increment
  // variables that provide levels information.
  EXPECT(dx2.variables().variables() == dx1.variables().variables());
}

// -----------------------------------------------------------------------------

template <typename MODEL>
class FieldSet4D : public oops::Test {
 public:
  FieldSet4D() = default;
  virtual ~FieldSet4D() = default;

 private:
  std::string testid() const override {return "test::FieldSet4D<" + MODEL::name() + ">";}

  void register_tests() const override {
    std::vector<eckit::testing::Test>& ts = eckit::testing::specification();

    ts.emplace_back(CASE("interface/FieldSet4D/testFieldSet3D")
      { testFieldSet3D<MODEL>(); });

    ts.emplace_back(CASE("interface/FieldSet4D/testFieldSet4D")
      { testFieldSet4D<MODEL>(); });
  }

  void clear() const override {}
};

// -----------------------------------------------------------------------------

}  // namespace test
