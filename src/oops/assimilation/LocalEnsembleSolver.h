/*
 * (C) Copyright 2020 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef OOPS_ASSIMILATION_LOCALENSEMBLESOLVER_H_
#define OOPS_ASSIMILATION_LOCALENSEMBLESOLVER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "eckit/config/Configuration.h"
#include "eckit/config/LocalConfiguration.h"
#include "oops/base/Departures.h"
#include "oops/base/DeparturesEnsemble.h"
#include "oops/base/IncrementEnsemble4D.h"
#include "oops/base/ObsAuxControls.h"
#include "oops/base/ObsEnsemble.h"
#include "oops/base/ObsErrors.h"
#include "oops/base/Observations.h"
#include "oops/base/Observers.h"
#include "oops/base/ObsLocalizations.h"
#include "oops/base/ObsSpaces.h"
#include "oops/base/StateEnsemble4D.h"
#include "oops/generic/PseudoModelState4D.h"
#include "oops/interface/Geometry.h"
#include "oops/interface/GeometryIterator.h"
#include "oops/interface/Model.h"
#include "oops/interface/ModelAuxControl.h"
#include "oops/interface/State.h"
#include "oops/util/ConfigFunctions.h"
#include "oops/util/Logger.h"
#include "oops/util/Timer.h"

namespace oops {

/// \brief Base class for LETKF-type solvers
template <typename MODEL, typename OBS>
class LocalEnsembleSolver {
  typedef Observers<MODEL, OBS>       Observers_;
  typedef Departures<OBS>             Departures_;
  typedef DeparturesEnsemble<OBS>     DeparturesEnsemble_;
  typedef Geometry<MODEL>             Geometry_;
  typedef GeometryIterator<MODEL>     GeometryIterator_;
  typedef IncrementEnsemble4D<MODEL>  IncrementEnsemble4D_;
  typedef ObsAuxControls<OBS>         ObsAux_;
  typedef ObsEnsemble<OBS>            ObsEnsemble_;
  typedef ObsErrors<OBS>              ObsErrors_;
  typedef Observations<OBS>           Observations_;
  typedef ObsLocalizations<MODEL, OBS> ObsLocalizations_;
  typedef ObsSpaces<OBS>              ObsSpaces_;
  typedef StateEnsemble4D<MODEL>      StateEnsemble4D_;
  typedef PseudoModelState4D<MODEL>   PseudoModel_;
  typedef State4D<MODEL>              State4D_;
  typedef State<MODEL>                State_;
  typedef Model<MODEL>                Model_;
  typedef ModelAuxControl<MODEL>      ModelAux_;
  typedef ObsDataVector<OBS, int>     ObsData_;
  typedef std::vector<std::shared_ptr<ObsData_>> ObsDataVec_;

 public:
  static const std::string classname() {return "oops::LocalEnsembleSolver";}

  /// initialize solver with \p obspaces, \p geometry, full \p config and \p nens ensemble size
  LocalEnsembleSolver(ObsSpaces_ & obspaces, const Geometry_ & geometry,
                      const eckit::Configuration & config, size_t nens);
  virtual ~LocalEnsembleSolver() = default;

  /// computes ensemble H(\p xx), returns mean H(\p xx), saves as hofx \p iteration
  virtual Observations_ computeHofX(const StateEnsemble4D_ & xx, size_t iteration,
                      bool readFromDisk);

  /// update background ensemble \p bg to analysis ensemble \p an at a grid point location \p i
  virtual void measurementUpdate(const IncrementEnsemble4D_ & bg,
                                 const GeometryIterator_ & i, IncrementEnsemble4D_ & an) = 0;

  /// copy \p an[\p i] = \p bg[\p i] (e.g. when there are no local observations to update state)
  virtual void copyLocalIncrement(const IncrementEnsemble4D_ & bg,
                                  const GeometryIterator_ & i, IncrementEnsemble4D_ & an) const;

  /// compute H(x) based on 4D state \p xx and put the result into \p yy. Also sets up
  /// R_ based on the QC filters run during H(x)
  void computeHofX4D(const eckit::Configuration &, const State4D_ &, Observations_ &);
  /// accessor to obs localizations
  const ObsLocalizations_ & obsloc() const {return obsloc_;}

 protected:
  const Geometry_  & geometry_;   ///< Geometry associated with the updated states
  const ObsSpaces_ & obspaces_;   ///< ObsSpaces used in the update
  Departures_ omb_;               ///< obs - mean(H(x)); set in computeHofX method
  DeparturesEnsemble_ Yb_;        ///< ensemble perturbations in the observation space;
                                  ///  set in computeHofX method
  std::unique_ptr<ObsErrors_>   R_;        ///< observation errors, set in computeHofX method
  std::unique_ptr<Departures_> invVarR_;   ///< inverse observation error variance; set in
                                           ///  computeHofX method

 private:
  const eckit::LocalConfiguration obsconf_;  // configuration for observations
  ObsLocalizations_ obsloc_;      ///< observation space localization
};

// -----------------------------------------------------------------------------

template <typename MODEL, typename OBS>
LocalEnsembleSolver<MODEL, OBS>::LocalEnsembleSolver(ObsSpaces_ & obspaces,
                                        const Geometry_ & geometry,
                                        const eckit::Configuration & config, size_t nens)
  : geometry_(geometry), obspaces_(obspaces), omb_(obspaces_), Yb_(obspaces_, nens),
    obsconf_(config, "observations"), obsloc_(obsconf_, obspaces_)
{
}

// -----------------------------------------------------------------------------

template <typename MODEL, typename OBS>
void LocalEnsembleSolver<MODEL, OBS>::computeHofX4D(const eckit::Configuration & config,
                                                    const State4D_ & xx, Observations_ & yy) {
  // compute forecast length from State4D times
  const std::vector<util::DateTime> times = xx.validTimes();
  const util::Duration flength = times[times.size()-1] - times[0];
  // observation window is passed to PseudoModel as the default model time step.
  // This is required when State4D has a single state: in this case if the default
  // model time step is not specified, it will be set to zero and no observations
  // will be processed during the "forecast"
  const util::Duration winlen = obspaces_.windowEnd() - obspaces_.windowStart();

  // Setup PseudoModelState4D
  std::unique_ptr<PseudoModel_> pseudomodel(new PseudoModel_(xx, winlen));
  const Model_ model(std::move(pseudomodel));
  // Setup model and obs biases; obs errors
  ModelAux_ moderr(geometry_, eckit::LocalConfiguration());
  ObsAux_ obsaux(obspaces_, obsconf_);
  R_.reset(new ObsErrors_(obsconf_, obspaces_));
  // Setup and run the model forecast with observers
  State_ init_xx = xx[0];
  PostProcessor<State_> post;
  Observers_ hofx(obspaces_, obsconf_);

  hofx.initialize(geometry_, obsaux, *R_, post, config);
  model.forecast(init_xx, moderr, flength, post);
  hofx.finalize(yy);
}

// -----------------------------------------------------------------------------

template <typename MODEL, typename OBS>
Observations<OBS> LocalEnsembleSolver<MODEL, OBS>::computeHofX(const StateEnsemble4D_ & ens_xx,
                                                   size_t iteration, bool readFromDisk) {
  util::Timer timer(classname(), "computeHofX");

  ASSERT(ens_xx.size() == Yb_.size());

  const size_t nens = ens_xx.size();
  ObsEnsemble_ obsens(obspaces_, nens);

  if (readFromDisk) {
    // read hofx from disk
    Log::debug() << "Read H(X) from disk" << std::endl;
    for (size_t jj = 0; jj < nens; ++jj) {
      obsens[jj].read("hofx"+std::to_string(iteration)+"_"+std::to_string(jj+1));
      Log::test() << "H(x) for member " << jj+1 << ":" << std::endl << obsens[jj] << std::endl;
    }
    R_.reset(new ObsErrors_(obsconf_, obspaces_));
  } else {
    // compute and save H(x)
    Log::debug() << "Computing H(X) online" << std::endl;
    for (size_t jj = 0; jj < nens; ++jj) {
      eckit::LocalConfiguration config;
      // never save H(x) (saved explicitly below), save EffectiveQC and EffectiveError
      // only for the last member
      config.set("save hofx", false);
      config.set("save qc", (jj == nens-1));
      config.set("save obs errors", (jj == nens-1));
      computeHofX4D(config, ens_xx[jj], obsens[jj]);
      Log::test() << "H(x) for member " << jj+1 << ":" << std::endl << obsens[jj] << std::endl;
      obsens[jj].save("hofx"+std::to_string(iteration)+"_"+std::to_string(jj+1));
    }
    // QC flags and Obs errors are set to that of the last ensemble member
    // TODO(someone) combine qc flags from all ensemble members
    R_->save("ObsError");
  }
  // set inverse variances
  invVarR_.reset(new Departures_(R_->inverseVariance()));

  // calculate H(x) ensemble mean
  Observations_ yb_mean(obsens.mean());

  // calculate H(x) ensemble perturbations
  for (size_t iens = 0; iens < nens; ++iens) {
    Yb_[iens] = obsens[iens] - yb_mean;
    Yb_[iens].mask(*invVarR_);
  }

  // calculate obs departures and mask with qc flag
  Observations_ yobs(obspaces_, "ObsValue");
  omb_ = yobs - yb_mean;
  omb_.mask(*invVarR_);

  // return mean H(x)
  return yb_mean;
}

// -----------------------------------------------------------------------------

template <typename MODEL, typename OBS>
void LocalEnsembleSolver<MODEL, OBS>::copyLocalIncrement(const IncrementEnsemble4D_ & bkg_pert,
                                                         const GeometryIterator_ & i,
                                                         IncrementEnsemble4D_ & ana_pert) const {
  // ana_pert[i]=bkg_pert[i]
  for (size_t itime=0; itime < bkg_pert[0].size(); ++itime) {
    for (size_t iens=0; iens < bkg_pert.size(); ++iens) {
      LocalIncrement gp = bkg_pert[iens][itime].getLocal(i);
      ana_pert[iens][itime].setLocal(gp, i);
    }
  }
}

// =============================================================================

/// \brief factory for LETKF solvers
template <typename MODEL, typename OBS>
class LocalEnsembleSolverFactory {
  typedef Geometry<MODEL>           Geometry_;
  typedef ObsSpaces<OBS>            ObsSpaces_;
 public:
  static std::unique_ptr<LocalEnsembleSolver<MODEL, OBS>> create(ObsSpaces_ &, const Geometry_ &,
                                                        const eckit::Configuration &,
                                                        size_t);
  virtual ~LocalEnsembleSolverFactory() = default;
 protected:
  explicit LocalEnsembleSolverFactory(const std::string &);
 private:
  virtual LocalEnsembleSolver<MODEL, OBS> * make(ObsSpaces_ &, const Geometry_ &,
                                        const eckit::Configuration &, size_t) = 0;
  static std::map < std::string, LocalEnsembleSolverFactory<MODEL, OBS> * > & getMakers() {
    static std::map < std::string, LocalEnsembleSolverFactory<MODEL, OBS> * > makers_;
    return makers_;
  }
};

// -----------------------------------------------------------------------------

template<class MODEL, class OBS, class T>
class LocalEnsembleSolverMaker : public LocalEnsembleSolverFactory<MODEL, OBS> {
  typedef Geometry<MODEL>           Geometry_;
  typedef ObsSpaces<OBS>            ObsSpaces_;

  virtual LocalEnsembleSolver<MODEL, OBS> * make(ObsSpaces_ & obspaces, const Geometry_ & geometry,
                                        const eckit::Configuration & conf, size_t nens)
    { return new T(obspaces, geometry, conf, nens); }
 public:
  explicit LocalEnsembleSolverMaker(const std::string & name)
    : LocalEnsembleSolverFactory<MODEL, OBS>(name) {}
};

// =============================================================================

template <typename MODEL, typename OBS>
LocalEnsembleSolverFactory<MODEL, OBS>::LocalEnsembleSolverFactory(const std::string & name) {
  if (getMakers().find(name) != getMakers().end()) {
    throw std::runtime_error(name + " already registered in local ensemble solver factory.");
  }
  getMakers()[name] = this;
}

// -----------------------------------------------------------------------------

template <typename MODEL, typename OBS>
std::unique_ptr<LocalEnsembleSolver<MODEL, OBS>>
LocalEnsembleSolverFactory<MODEL, OBS>::create(ObsSpaces_ & obspaces, const Geometry_ & geometry,
                                  const eckit::Configuration & conf, size_t nens) {
  Log::trace() << "LocalEnsembleSolver<MODEL, OBS>::create starting" << std::endl;
  const std::string id = conf.getString("local ensemble DA.solver");
  typename std::map<std::string, LocalEnsembleSolverFactory<MODEL, OBS>*>::iterator
    jloc = getMakers().find(id);
  if (jloc == getMakers().end()) {
    Log::error() << id << " does not exist in local ensemble solver factory." << std::endl;
    Log::error() << "LETKF solver Factory has " << getMakers().size() << " elements:" << std::endl;
    for (typename std::map<std::string, LocalEnsembleSolverFactory<MODEL, OBS>*>::const_iterator
         jj = getMakers().begin(); jj != getMakers().end(); ++jj) {
       Log::error() << "A " << jj->first << " LocalEnsembleSolver" << std::endl;
    }
    throw std::runtime_error(id + " does not exist in local ensemble solver factory.");
  }
  std::unique_ptr<LocalEnsembleSolver<MODEL, OBS>>
    ptr(jloc->second->make(obspaces, geometry, conf, nens));
  Log::trace() << "LocalEnsembleSolver<MODEL, OBS>::create done" << std::endl;
  return ptr;
}

// -----------------------------------------------------------------------------

}  // namespace oops
#endif  // OOPS_ASSIMILATION_LOCALENSEMBLESOLVER_H_
