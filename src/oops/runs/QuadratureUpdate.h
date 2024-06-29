/*
 * (C) Copyright 2024 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef OOPS_RUNS_QUADRATUREUPDATE_H_
#define OOPS_RUNS_QUADRATUREUPDATE_H_

#include <memory>
#include <string>

#include "eckit/config/LocalConfiguration.h"
#include "oops/assimilation/ControlVariable.h"
#include "oops/assimilation/ControlIncrement.h"
#include "oops/assimilation/DualVector.h"
#include "oops/assimilation/CostFunction.h"
#include "oops/assimilation/IncrementalAssimilation.h"
#include "oops/assimilation/instantiateCostFactory.h"
#include "oops/assimilation/instantiateMinFactory.h"
#include "oops/assimilation/HMatrix.h"
#include "oops/assimilation/RinvSqrtMatrix.h"
#include "oops/assimilation/NormalizedHBHtMatrix.h"
#include "oops/assimilation/QuadratureSolver.h"
#include "oops/assimilation/SLCG.h"
#include "oops/base/Geometry.h"
#include "oops/base/Increment.h"
#include "oops/base/instantiateCovarFactory.h"
#include "oops/base/instantiateObsFilterFactory.h"
#include "oops/base/PostProcessor.h"
#include "oops/base/State.h"
#include "oops/base/State4D.h"
#include "oops/base/StateInfo.h"
#include "oops/base/StateWriter.h"
#include "oops/base/StructuredGridPostProcessor.h"
#include "oops/base/StructuredGridWriter.h"
#include "oops/generic/instantiateLinearModelFactory.h"
#include "oops/generic/instantiateNormFactory.h"
#include "oops/generic/instantiateObsErrorFactory.h"
#include "oops/mpi/mpi.h"
#include "oops/runs/Application.h"
#include "oops/util/DateTime.h"
#include "oops/util/Logger.h"
#include "oops/util/parameters/Parameter.h"
#include "oops/util/parameters/Parameters.h"
#include "oops/util/parameters/RequiredParameter.h"
#include "oops/util/printRunStats.h"

namespace oops {

template <typename MODEL, typename OBS>
class QuadratureUpdateParameters : public ApplicationParameters {
  OOPS_CONCRETE_PARAMETERS(QuadratureUpdateParameters, ApplicationParameters)

 public:
  /// Parameters for ensemble member to be updated.
  RequiredParameter<eckit::LocalConfiguration> ensMemberConfig{"ensemble state", this};

  /// Parameters for variational assimilation
  RequiredParameter<eckit::LocalConfiguration> varConfig{"variational", this};

  /// Parameters for cost function used in initialization
  RequiredParameter<eckit::LocalConfiguration> finalConfig{"final", this};

  /// Parameters for cost function used in initialization
  RequiredParameter<eckit::LocalConfiguration> cfConfig{"cost function", this};
};

template <typename MODEL, typename OBS> class QuadratureUpdate : public Application {
  typedef Geometry<MODEL>                   Geometry_;
  typedef Increment<MODEL>                  Increment_;
  typedef ControlVariable<MODEL, OBS>       CtrlVar_;
  typedef ControlIncrement<MODEL, OBS>      CtrlInc_;
  typedef DualVector<MODEL, OBS>            Dual_;
  typedef State<MODEL>                      State_;
  typedef State4D<MODEL>                    State4D_;
  typedef Model<MODEL>                      Model_;
  typedef ModelAuxControl<MODEL>            ModelAux_;
  typedef ObsAuxControls<OBS>               ObsAux_;
  typedef CostJbTotal<MODEL, OBS>           JbTotal_;
  typedef HMatrix<MODEL, OBS>               H_;
  typedef RinvSqrtMatrix<MODEL, OBS>        R_invsqrt_;
  typedef NormalizedHBHtMatrix<MODEL, OBS>  NormalHBHt_;
  typedef QuadratureSolver<MODEL, OBS>      QuadSolver_;

  typedef QuadratureUpdateParameters<MODEL, OBS> QuadParams_;

 public:
// -----------------------------------------------------------------------------
  explicit QuadratureUpdate(const eckit::mpi::Comm & comm = oops::mpi::world()) : Application(comm) {
    instantiateCostFactory<MODEL, OBS>();
    instantiateCovarFactory<MODEL>();
    instantiateMinFactory<MODEL, OBS>();
    instantiateNormFactory<MODEL>();
    instantiateObsErrorFactory<OBS>();
    instantiateObsFilterFactory<OBS>();
    instantiateLinearModelFactory<MODEL>();
  }
// -----------------------------------------------------------------------------
  virtual ~QuadratureUpdate() {}
// -----------------------------------------------------------------------------
  int execute(const eckit::Configuration & fullConfig, bool validate) const override {
    Log::trace() << "QuadratureUpdate: execute start" << std::endl;
    util::printRunStats("QuadratureUpdate start");

//  Deserialize parameters
    QuadParams_ params;
    if (validate) params.validate(fullConfig);
    params.deserialize(fullConfig);

//  Setup outer loop
    eckit::LocalConfiguration varConf(fullConfig, "variational");
    std::vector<eckit::LocalConfiguration> iterconfs;
    varConf.get("iterations", iterconfs);

/// The background is constructed inside the cost function because its valid
/// time within the assimilation window can be different (3D-Var vs. 4D-Var),
/// it can be 3D or 4D (strong vs weak constraint), etc...

//  Setup cost function
    std::unique_ptr<CostFunction<MODEL, OBS>>
      J(CostFactory<MODEL, OBS>::create(params.cfConfig, this->getComm()));
    const JbTotal_ & Jb = J->jb();

//  Get the forecast mean and model auxiliary information
    CtrlVar_ x_bg(Jb.getBackground());
    ModelAux_ & maux = x_bg.modVar();
    ObsAux_ & oaux   = x_bg.obsVar();

//  Setup post-processor
    PostProcessor<State_> post;
    if (iterconfs[0].has("prints")) {
      const eckit::LocalConfiguration prtConfig(iterconfs[0], "prints");
      post.enrollProcessor(new StateInfo<State_>("traj", prtConfig));
    }

/// "Dummy" evaluation of test function, which serves to properly initialize
/// the geometry information. Without this, the program will segfault upon
/// trying to initialize a ControlIncrement.

    iterconfs[0].set("linearize", true);
    J->evaluate(x_bg, iterconfs[0], post);
    util::printRunStats("QuadratureUpdate linearize " + std::to_string(0));

//  Setup independent geometry configuration object for reading ensemble members
    const Geometry_ geometry(params.cfConfig.value().getSubConfiguration("geometry"), this->getComm(), mpi::myself());

//  Get the ensemble state to be updated
    State4D_ x_ens_state(geometry, params.ensMemberConfig);
    Log::test() << "Ensemble state: " << x_ens_state << std::endl;

//  Wrapping the ensemble state in a control variable
    std::shared_ptr<State4D_>  xens_ptr(&x_ens_state);
    std::shared_ptr<ModelAux_> maux_ptr(&maux);
    std::shared_ptr<ObsAux_>   oaux_ptr(&oaux);
    CtrlVar_                   x_ens(xens_ptr, maux_ptr, oaux_ptr);

//  Taking difference between background and ensemble state to form the increment that will be transformed.
    CtrlInc_ dx_in(Jb);
    dx_in.diff(x_ens, x_bg);

    QuadSolver_ quadsolver(*J);
    CtrlInc_ dx_out = quadsolver.solve(dx_in, 20, 100, 1e-2);

    std::cout << "DEBUGGING: Finished quadrature solve." << std::endl;

//     std::cout << "DEBUGGING: normalizing by obs-error covariance." << std::endl;
// //  Normalizing by the obs-error covariance.
//     Dual_ dw_pre;
//     R_invsqrt_mat.multiply(dy, dw_pre);

//     std::cout << "DEBUGGING: running SLCG." << std::endl;
//     std::vector<Dual_> X;
//     std::vector<double> lambda; lambda.push_back(1);
//     SLCG(X, NormalHBHt_mat, dw_pre, lambda, 1, 20, 1e-10);
    

// //  Perform Incremental Variational Assimilation
//     eckit::LocalConfiguration varConf(fullConfig, "variational");
//     int iouter = IncrementalAssimilation<MODEL, OBS>(xx, *J, varConf);
//     Log::info() << "QuadratureUpdate: incremental assimilation done "
//                 << iouter << " iterations." << std::endl;

// //  Save analysis and final diagnostics
//     PostProcessor<State_> post;
//     if (fullConfig.has("variational output")) {
//       const eckit::LocalConfiguration outConfig(fullConfig, "variational output");
//       post.enrollProcessor(new StateWriter<State_>(outConfig));
//     }

//     eckit::LocalConfiguration finalConfig(fullConfig, "final");
//     finalConfig.set("iteration", iouter);

// //  Save increment if desired
//     if (finalConfig.has("increment")) {
//       const eckit::LocalConfiguration incConfig(finalConfig, "increment");
//       ControlVariable<MODEL, OBS> x_b(J->jb().getBackground());
//       const eckit::LocalConfiguration incGeomConfig(incConfig, "geometry");
//       Geometry<MODEL> incGeom(incGeomConfig,
//                               xx.states().geometry().getComm(),
//                               xx.states().commTime());
//       ControlIncrement<MODEL, OBS> dx_tmp(J->jb());
//       ControlIncrement<MODEL, OBS> dx(incGeom, dx_tmp);
//       dx.diff(xx, x_b);
//       const eckit::LocalConfiguration incOutConfig(incConfig, "variational output");
//       dx.write(incOutConfig);
//     }

//     if (finalConfig.has("increment to structured grid")) {
//       const eckit::LocalConfiguration incLatlonConf(finalConfig, "increment to structured grid");

//       ControlVariable<MODEL, OBS> x_b(J->jb().getBackground());
//       ControlIncrement<MODEL, OBS> dx(J->jb());
//       dx.diff(xx, x_b);

//       const StructuredGridWriter<MODEL> latlon(incLatlonConf, dx.states().geometry());
//       for (size_t jtime = 0; jtime < dx.states().size(); ++jtime) {
//         latlon.interpolateAndWrite(dx.states()[jtime], xx.states()[jtime]);
//       }
//     }

//     if (finalConfig.has("prints")) {
//       const eckit::LocalConfiguration prtConfig(finalConfig, "prints");
//       post.enrollProcessor(new StateInfo<State_>("final", prtConfig));
//     }

//     if (finalConfig.has("analysis to structured grid")) {
//       const eckit::LocalConfiguration anLatlonConf(finalConfig, "analysis to structured grid");
//       post.enrollProcessor(new StructuredGridPostProcessor<MODEL, State_>(
//             anLatlonConf, xx.state().geometry() ));
//     }

//     J->evaluate(xx, finalConfig, post);

// //  Save ObsAux
//     xx.obsVar().write(cfConf);

//     if (finalConfig.has("forecast from analysis")) {
//       const eckit::LocalConfiguration fcFromAnConf(finalConfig, "forecast from analysis");

//       //  Setup Model
//       const Model_ model(xx.state().geometry(), eckit::LocalConfiguration(fcFromAnConf, "model"));

//       //  Setup augmented state
//       const ModelAux_ moderr(xx.state().geometry(),
//                             eckit::LocalConfiguration(fcFromAnConf, "model aux control"));

//       //  Setup times
//       const util::Duration fclength(fcFromAnConf.getString("forecast length"));
//       const util::DateTime bgndate(xx.state().validTime());
//       const util::DateTime enddate(bgndate + fclength);
//       Log::info() << "Running forecast from " << bgndate << " to " << enddate << std::endl;

//       //  Setup forecast outputs
//       PostProcessor<State_> post;

//       eckit::LocalConfiguration prtConfig;
//       if (fcFromAnConf.has("prints")) {
//         prtConfig = eckit::LocalConfiguration(fcFromAnConf, "prints");
//         post.enrollProcessor(new StateInfo<State_>("fc", prtConfig));
//       }

//       eckit::LocalConfiguration outConfig;
//       if (fcFromAnConf.has("variational output")) {
//         outConfig = eckit::LocalConfiguration(fcFromAnConf, "variational output");
//         outConfig.set("date", bgndate.toString());
//         post.enrollProcessor(new StateWriter<State_>(outConfig));
//       }
//       //  Run forecast

//       Log::test() << "Inital state: " << xx.state() << std::endl;

//       model.forecast(xx.state(), moderr, fclength, post);

//       Log::test() << "Final state: " << xx.state() << std::endl;
//     }

    util::printRunStats("QuadratureUpdate end");
    Log::trace() << "QuadratureUpdate: execute done" << std::endl;
    return 0;
  }
// -----------------------------------------------------------------------------
  void validateConfig(const eckit::Configuration & fullConfig) const override {
    // Note: Variational app doesn't have application level Parameters yet;
    // not validating anything.
  }
// -----------------------------------------------------------------------------
 private:
  std::string appname() const override {
    return "oops::QuadratureUpdate<" + MODEL::name() + ", " + OBS::name() + ">";
  }
// -----------------------------------------------------------------------------
};

}  // namespace oops
#endif  // OOPS_RUNS_VARIATIONAL_H_
