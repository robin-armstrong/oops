/*
 * (C) Copyright 2009-2016 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef OOPS_RUNS_HOFX_H_
#define OOPS_RUNS_HOFX_H_

#include <string>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "oops/assimilation/CalcHofX.h"
#include "oops/base/instantiateObsFilterFactory.h"
#include "oops/base/Observations.h"
#include "oops/base/ObsSpaces.h"
#include "oops/base/PostProcessor.h"
#include "oops/base/StateInfo.h"
#include "oops/interface/Geometry.h"
#include "oops/interface/Model.h"
#include "oops/interface/State.h"
#include "oops/parallel/mpi/mpi.h"
#include "oops/runs/Application.h"
#include "oops/util/DateTime.h"
#include "oops/util/Duration.h"
#include "oops/util/Logger.h"

namespace oops {

template <typename MODEL> class HofX : public Application {
  typedef Geometry<MODEL>            Geometry_;
  typedef Model<MODEL>               Model_;
  typedef Observations<MODEL>        Observations_;
  typedef ObsSpaces<MODEL>           ObsSpaces_;
  typedef State<MODEL>               State_;

 public:
// -----------------------------------------------------------------------------
  explicit HofX(const eckit::mpi::Comm & comm = oops::mpi::comm()) : Application(comm) {
    instantiateObsFilterFactory<MODEL>();
  }
// -----------------------------------------------------------------------------
  virtual ~HofX() {}
// -----------------------------------------------------------------------------
  int execute(const eckit::Configuration & fullConfig) const {
//  Setup observation window
    const eckit::LocalConfiguration windowConf(fullConfig, "Assimilation Window");
    const util::Duration winlen(windowConf.getString("window_length"));
    const util::DateTime winbgn(windowConf.getString("window_begin"));
    const util::DateTime winend(winbgn + winlen);
    Log::info() << "Observation window is:" << windowConf << std::endl;

//  Setup geometry
    const eckit::LocalConfiguration geometryConfig(fullConfig, "Geometry");
    const Geometry_ geometry(geometryConfig, this->getComm());

//  Setup Model
    const eckit::LocalConfiguration modelConfig(fullConfig, "Model");
    const Model_ model(geometry, modelConfig);

//  Setup initial state
    const eckit::LocalConfiguration initialConfig(fullConfig, "Initial Condition");
    State_ xx(geometry, model.variables(), initialConfig);
    Log::test() << "Initial state: " << xx << std::endl;

//  Setup forecast outputs
    PostProcessor<State_> post;

    eckit::LocalConfiguration prtConf;
    fullConfig.get("Prints", prtConf);
    post.enrollProcessor(new StateInfo<State_>("fc", prtConf));

//  Setup observations
    const eckit::LocalConfiguration obsconf(fullConfig, "Observations");
    Log::info() << "Observations configuration is:" << obsconf << std::endl;
    ObsSpaces_ obspace(obsconf, this->getComm(), winbgn, winend);

//  Setup and run observer
    CalcHofX<MODEL> hofx(obspace, geometry, fullConfig);
    const Observations_ & yobs = hofx.compute(model, xx, post);
    for (size_t jj = 0; jj < obspace.size(); ++jj) {
      hofx.qcFlags(jj).save("EffectiveQC");
      hofx.obsErrors(jj).save("EffectiveError");
    }

    Log::test() << "Final state: " << xx << std::endl;

//  Save H(x)
    Log::test() << "H(x): " << std::endl << yobs << "End H(x)" << std::endl;
    yobs.save("hofx");

    return 0;
  }
// -----------------------------------------------------------------------------
 private:
  std::string appname() const {
    return "oops::HofX<" + MODEL::name() + ">";
  }
// -----------------------------------------------------------------------------
};

}  // namespace oops

#endif  // OOPS_RUNS_HOFX_H_
