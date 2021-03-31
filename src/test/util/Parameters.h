/*
 * (C) Copyright 2019 Met Office UK
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef TEST_UTIL_PARAMETERS_H_
#define TEST_UTIL_PARAMETERS_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <boost/make_unique.hpp>

#define ECKIT_TESTING_SELF_REGISTER_CASES 0

#include "eckit/config/LocalConfiguration.h"
#include "eckit/testing/Test.h"
#include "oops/../test/TestEnvironment.h"
#include "oops/base/ParameterTraitsVariables.h"
#include "oops/base/Variables.h"
#include "oops/runs/Test.h"
#include "oops/util/AnyOf.h"
#include "oops/util/Expect.h"
#include "oops/util/Logger.h"
#include "oops/util/parameters/ConfigurationParameter.h"
#include "oops/util/parameters/HasParameters_.h"
#include "oops/util/parameters/IgnoreOtherParameters.h"
#include "oops/util/parameters/NumericConstraints.h"
#include "oops/util/parameters/OptionalParameter.h"
#include "oops/util/parameters/OptionalPolymorphicParameter.h"
#include "oops/util/parameters/Parameter.h"
#include "oops/util/parameters/Parameters.h"
#include "oops/util/parameters/ParameterTraits.h"
#include "oops/util/parameters/ParameterTraitsAnyOf.h"
#include "oops/util/parameters/ParameterTraitsScalarOrMap.h"
#include "oops/util/parameters/PolymorphicParameter.h"
#include "oops/util/parameters/RequiredParameter.h"
#include "oops/util/parameters/RequiredPolymorphicParameter.h"

namespace test {

enum class Fruit {
  APPLE, ORANGE
};

struct FruitParameterTraitsHelper {
  typedef Fruit EnumType;
  static constexpr char enumTypeName[] = "Fruit";
  static constexpr util::NamedEnumerator<Fruit> namedValues[] = {
    { Fruit::APPLE, "apple" },
    { Fruit::ORANGE, "orange" }
  };
};

constexpr char FruitParameterTraitsHelper::enumTypeName[];
constexpr util::NamedEnumerator<Fruit> FruitParameterTraitsHelper::namedValues[];

const bool validationSupported = oops::Parameters::isValidationSupported();

}  // namespace test

namespace oops {

template <>
struct ParameterTraits<test::Fruit> : public EnumParameterTraits<test::FruitParameterTraitsHelper>
{};

}  // namespace oops

namespace test {

// Classes required by most tests in this file

class RangeParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(RangeParameters, Parameters)
 public:
  oops::Parameter<float> minParameter{"min", 0.0f, this};
  oops::Parameter<float> maxParameter{"max", 0.0f, this};
};

class EmbeddedParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(EmbeddedParameters, Parameters)
 public:
  oops::Parameter<int> intParameter{"embedded_int_parameter", 3, this};
  oops::OptionalParameter<util::DateTime> optDateTimeParameter{"opt_embedded_date_time_parameter",
                                                               this};
};

class MyParametersBase : public oops::Parameters {
  OOPS_ABSTRACT_PARAMETERS(MyParametersBase, Parameters)
 public:
  typedef util::AnyOf<std::string, std::vector<int>> AnyOf_;

  oops::Parameter<float> floatParameter{"float_parameter", 1.5f, this};
  oops::Parameter<int> intParameter{"int_parameter", 2, this};
  oops::Parameter<bool> boolParameter{"bool_parameter", true, this};
  oops::OptionalParameter<float> optFloatParameter{"opt_float_parameter", this};
  oops::OptionalParameter<util::DateTime> optDateTimeParameter{"opt_date_time_parameter", this};
  oops::OptionalParameter<util::Duration> optDurationParameter{"opt_duration_parameter", this};
  oops::OptionalParameter<util::PartialDateTime> optPartialDateTimeParameter{
    "opt_partialDT_parameter", this};
  oops::Parameter<Fruit> fruitParameter{"fruit_parameter", Fruit::ORANGE, this};
  oops::Parameter<RangeParameters> rangeParameter{"range_parameter", RangeParameters(), this};
  oops::Parameter<std::vector<int>> intParameters{"int_parameters", {}, this};
  oops::Parameter<std::vector<RangeParameters>> rangeParameters{"range_parameters", {}, this};
  oops::Parameter<oops::Variables> variablesParameter{"variables_parameter", {}, this};
  oops::Parameter<std::set<int>> setIntParameter{"set_int_parameter", {}, this};
  oops::Parameter<AnyOf_> anyOfParameter{
    "any_of_parameter", AnyOf_(std::vector<int>({10, 20})), this};
  oops::OptionalParameter<AnyOf_> optAnyOfParameter{"opt_any_of_parameter", this};
  oops::OptionalParameter<void> optNullParameter{"opt_null_parameter", this};
  EmbeddedParameters embeddedParameters{this};
};

class MyOptionalParameters : public MyParametersBase {
  OOPS_CONCRETE_PARAMETERS(MyOptionalParameters, MyParametersBase)
 public:
};

class MyOptionalAndRequiredParameters : public MyParametersBase {
  OOPS_CONCRETE_PARAMETERS(MyOptionalAndRequiredParameters, MyParametersBase)
 public:
  oops::RequiredParameter<float> reqFloatParameter{"req_float_parameter", this};
  oops::RequiredParameter<util::Duration> reqDurationParameter{"req_duration_parameter", this};
};

// Class required by tests checking map-valued parameters

class MyMapParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(MyMapParameters, Parameters)
 public:
  oops::Parameter<std::map<int, float>> intToFloatMapParameter{
      "int_to_float_map", std::map<int, float>(), this};
  oops::Parameter<std::map<std::string, util::Duration>> stringToDurationMapParameter{
      "string_to_duration_map", std::map<std::string, util::Duration>(), this};

  oops::Parameter<util::ScalarOrMap<int, float>> floatOrIntToFloatMapParameter1{
      "float_or_int_to_float_map_1", util::ScalarOrMap<int, float>(), this};
  oops::Parameter<util::ScalarOrMap<std::string, util::Duration>>
    durationOrStringToDurationMapParameter1{"duration_or_string_to_duration_map_1",
                                            util::ScalarOrMap<std::string, util::Duration>(), this};

  oops::Parameter<util::ScalarOrMap<int, float>> floatOrIntToFloatMapParameter2{
      "float_or_int_to_float_map_2", util::ScalarOrMap<int, float>(), this};
  oops::Parameter<util::ScalarOrMap<std::string, util::Duration>>
    durationOrStringToDurationMapParameter2{"duration_or_string_to_duration_map_2",
                                            util::ScalarOrMap<std::string, util::Duration>(), this};
};

// Class required by tests checking Variables-valued parameters

class VariablesParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(VariablesParameters, Parameters)
 public:
  oops::Parameter<oops::Variables> filterVariables{"filter_variables", {}, this};
  oops::Parameter<oops::Variables> operatorVariables{"operator_variables", {}, this};
};

// Classes required to test support for polymorphic parameters.

class DeviceTypeDependentParameters : public oops::Parameters {
  OOPS_ABSTRACT_PARAMETERS(DeviceTypeDependentParameters, Parameters)
};

class ScreenParameters : public DeviceTypeDependentParameters {
  OOPS_CONCRETE_PARAMETERS(ScreenParameters, DeviceTypeDependentParameters)
 public:
  oops::OptionalParameter<float> diameter{"diameter", this};
};

class PrinterParameters : public DeviceTypeDependentParameters {
  OOPS_CONCRETE_PARAMETERS(PrinterParameters, DeviceTypeDependentParameters)
 public:
  oops::Parameter<std::string> paperFormat{"paper_format", "A4", this};
};

class DeviceFactory {
 public:
  static std::unique_ptr<DeviceTypeDependentParameters> createParameters(const std::string &name) {
    if (name == "screen") {
      return boost::make_unique<ScreenParameters>();
    }
    if (name == "printer") {
      return boost::make_unique<PrinterParameters>();
    }
    throw eckit::BadParameter("Unrecognized device type");
  }

  static std::vector<std::string> getMakerNames() {
    return {"screen", "printer"};
  }
};

class RequiredDeviceParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(RequiredDeviceParameters, Parameters)
 public:
  oops::RequiredPolymorphicParameter<DeviceTypeDependentParameters, DeviceFactory>
    device{"type", this};
};

class DeviceParametersWithDefault : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(DeviceParametersWithDefault, Parameters)
 public:
  oops::PolymorphicParameter<DeviceTypeDependentParameters, DeviceFactory>
    device{"type", "screen", this};
};

class OptionalDeviceParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(OptionalDeviceParameters, Parameters)
 public:
  oops::OptionalPolymorphicParameter<DeviceTypeDependentParameters, DeviceFactory>
   device{"type", this};
};

class AllDeviceParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(AllDeviceParameters, Parameters)
 public:
  oops::RequiredParameter<RequiredDeviceParameters> requiredDevice{"required_device", this};
  oops::Parameter<DeviceParametersWithDefault> deviceWithDefault{"device_with_default", {}, this};
  oops::Parameter<OptionalDeviceParameters> optionalDevice{"optional_device", {}, this};
};

// Classes used to test parameter constraints

class ConstrainedParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(ConstrainedParameters, Parameters)
 public:
  oops::Parameter<int> intWithMin{
    "int_with_min", 10, this, {oops::minConstraint(5)}};
  oops::Parameter<int> intWithExclusiveMin{
    "int_with_exclusive_min", 10, this, {oops::exclusiveMinConstraint(5)}};
  oops::Parameter<int> intWithMax{
    "int_with_max", 0, this, {oops::maxConstraint(5)}};
  oops::Parameter<int> intWithExclusiveMax{
    "int_with_exclusive_max", 0, this, {oops::exclusiveMaxConstraint(5)}};
  oops::Parameter<float> floatWithMin{
    "float_with_min", 10.f, this, {oops::minConstraint(5.5f)}};
  oops::Parameter<float> floatWithExclusiveMin{
    "float_with_exclusive_min", 10.f, this, {oops::exclusiveMinConstraint(5.5f)}};
  oops::Parameter<float> floatWithMax{
    "float_with_max", 0.f, this, {oops::maxConstraint(5.5f)}};
  oops::Parameter<float> floatWithExclusiveMax{
    "float_with_exclusive_max", 0.f, this, {oops::exclusiveMaxConstraint(5.5f)}};
};

class ConstrainedRequiredParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(ConstrainedRequiredParameters, Parameters)
 public:
  oops::RequiredParameter<int> intWithMin{
    "int_with_min", this, {oops::minConstraint(5)}};
  oops::RequiredParameter<int> intWithExclusiveMin{
    "int_with_exclusive_min", this, {oops::exclusiveMinConstraint(5)}};
  oops::RequiredParameter<int> intWithMax{
    "int_with_max", this, {oops::maxConstraint(5)}};
  oops::RequiredParameter<int> intWithExclusiveMax{
    "int_with_exclusive_max", this, {oops::exclusiveMaxConstraint(5)}};
  oops::RequiredParameter<float> floatWithMin{
    "float_with_min", this, {oops::minConstraint(5.5f)}};
  oops::RequiredParameter<float> floatWithExclusiveMin{
    "float_with_exclusive_min", this, {oops::exclusiveMinConstraint(5.5f)}};
  oops::RequiredParameter<float> floatWithMax{
    "float_with_max", this, {oops::maxConstraint(5.5f)}};
  oops::RequiredParameter<float> floatWithExclusiveMax{
    "float_with_exclusive_max", this, {oops::exclusiveMaxConstraint(5.5f)}};
};

class ConstrainedOptionalParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(ConstrainedOptionalParameters, Parameters)
 public:
  oops::OptionalParameter<int> intWithMin{
    "int_with_min", this, {oops::minConstraint(5)}};
  oops::OptionalParameter<int> intWithExclusiveMin{
    "int_with_exclusive_min", this, {oops::exclusiveMinConstraint(5)}};
  oops::OptionalParameter<int> intWithMax{
    "int_with_max", this, {oops::maxConstraint(5)}};
  oops::OptionalParameter<int> intWithExclusiveMax{
    "int_with_exclusive_max", this, {oops::exclusiveMaxConstraint(5)}};
  oops::OptionalParameter<float> floatWithMin{
    "float_with_min", this, {oops::minConstraint(5.5f)}};
  oops::OptionalParameter<float> floatWithExclusiveMin{
    "float_with_exclusive_min", this, {oops::exclusiveMinConstraint(5.5f)}};
  oops::OptionalParameter<float> floatWithMax{
    "float_with_max", this, {oops::maxConstraint(5.5f)}};
  oops::OptionalParameter<float> floatWithExclusiveMax{
    "float_with_exclusive_max", this, {oops::exclusiveMaxConstraint(5.5f)}};
};

// Classes used to test the ConfigurationParameter class

class IncompleteParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(IncompleteParameters, Parameters)
 public:
  oops::Parameter<float> floatParameter{"float_parameter", 1.5f, this};
  oops::Parameter<int> intParameter{"int_parameter", 2, this};
  oops::ConfigurationParameter config{this};
};

// Classes used to test if conflicts between schemas are detected

class UnconstrainedParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(UnconstrainedParameters, Parameters)
 public:
  // This parameter's name matches the name of one of parameters in ConstrainedParameters,
  // but it doesn't specify a minimum constraint.
  oops::Parameter<int> intWithMin{"int_with_min", 10, this};
  // This parameter's name doesn't match the name of any parameter in ConstrainedParameters.
  oops::Parameter<int> height{"height", 10, this};
};

class OtherConstrainedParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(OtherConstrainedParameters, Parameters)
 public:
  // This parameter's name matches the name of one of parameters in ConstrainedParameters,
  // but it specifies a different minimum constraint.
  oops::Parameter<int> intWithMin{"int_with_min", 10, this, {oops::minConstraint(6)}};
};

// A mixture of constrained and unconstrained parameters, some of which have the same names
class ConstrainedThenUnconstrainedParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(ConstrainedThenUnconstrainedParameters, Parameters)
 public:
  ConstrainedParameters constrained{this};
  UnconstrainedParameters unconstrained{this};
};

// A mixture of unconstrained and constrained parameters, some of which have the same names
class UnconstrainedThenConstrainedParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(UnconstrainedThenConstrainedParameters, Parameters)
 public:
  UnconstrainedParameters unconstrained{this};
  ConstrainedParameters constrained{this};
};

// A mixture of two parameters classes, some of which have the same names, but specify different
// constraints
class ConflictingParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(ConflictingParameters, Parameters)
 public:
  ConstrainedParameters constrained{this};
  OtherConstrainedParameters otherConstrained{this};
};

// Classes used to test the IgnoreOtherParameters class

class TolerantParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(TolerantParameters, Parameters)
 public:
  oops::Parameter<float> floatParameter{"float_parameter", 1.5f, this};
  oops::Parameter<int> intParameter{"int_parameter", 2, this};
  oops::IgnoreOtherParameters config{this};
};

// Classes used to test HasParameters_

struct WithoutParameters_ {};

struct WithParameters_NotDerivedFromOopsParameters {
  typedef std::string Parameters_;
};

class PrivateParameters : private oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(PrivateParameters, Parameters)
 public:
};

struct WithParameters_DerivedPrivatelyFromOopsParameters {
  typedef PrivateParameters Parameters_;
};

namespace nonoops {

struct Parameters {};

struct SomeParameters : Parameters {};

}  // namespace nonoops

struct WithParameters_DerivedFromNonOopsParameters {
  typedef PrivateParameters Parameters_;
};

struct WithParameters_DerivedFromOopsParameters {
  typedef MyOptionalParameters Parameters_;
};


template <typename ParametersType>
void doTestSerialization(const eckit::Configuration &config) {
  // We deserialize a configuration loaded from a YAML file into parameters and then serialize them
  // back into a configuration. The test verifies that the configuration objects produce the same
  // output when printed.
  //
  // For this to work, parameter names in the YAML file must be ordered alphabetically; that's
  // because the YAML parser creates configurations storing keys and values OrderedMapContent
  // objects (preserving the order in which individual options were specified in the YAML file),
  // but the LocalConfiguration::set() method stores keys and values in MapContent objects (with
  // keys ordered alphabetically).

  ParametersType params;
  EXPECT_NO_THROW(params.validate(config));
  params.deserialize(config);

  eckit::LocalConfiguration outputConfig;
  params.serialize(outputConfig);

  std::stringstream expectedStream;
  expectedStream << config;
  const std::string expected = expectedStream.str();

  std::stringstream receivedStream;
  receivedStream << outputConfig;
  std::string received = receivedStream.str();

  EXPECT_EQUAL(received, expected);
}

void testDefaultValues() {
  const eckit::LocalConfiguration conf(TestEnvironment::config());

  MyOptionalAndRequiredParameters params;

  EXPECT_EQUAL(params.floatParameter, 1.5f);
  EXPECT_EQUAL(params.floatParameter.value(), 1.5f);
  EXPECT_EQUAL(params.intParameter, 2);
  EXPECT(params.boolParameter);
  EXPECT(params.optFloatParameter.value() == boost::none);
  EXPECT(params.optDateTimeParameter.value() == boost::none);
  EXPECT(params.optDurationParameter.value() == boost::none);
  EXPECT(params.optPartialDateTimeParameter.value() == boost::none);
  EXPECT_THROWS_AS(params.reqFloatParameter.value(), boost::bad_optional_access);
  EXPECT_THROWS_AS(params.reqDurationParameter.value(), boost::bad_optional_access);
  EXPECT(params.fruitParameter == Fruit::ORANGE);
  EXPECT(params.rangeParameter.value().minParameter == 0.0f);
  EXPECT(params.rangeParameter.value().maxParameter == 0.0f);
  EXPECT(params.intParameters.value().empty());
  EXPECT(params.rangeParameters.value().empty());
  EXPECT(params.variablesParameter.value() == oops::Variables());
  EXPECT(params.setIntParameter.value() == std::set<int>());
  EXPECT(params.anyOfParameter.value().as<std::vector<int>>() == std::vector<int>({10, 20}));
  EXPECT(params.optAnyOfParameter.value() == boost::none);
  EXPECT_NOT(params.optNullParameter.value());
  EXPECT(params.embeddedParameters.intParameter.value() == 3);
  EXPECT(params.embeddedParameters.optDateTimeParameter.value() == boost::none);

  const eckit::LocalConfiguration minimalConf(conf, "minimal");
  EXPECT_NO_THROW(params.validate(minimalConf));
  params.deserialize(minimalConf);

  EXPECT_EQUAL(params.floatParameter, 1.5f);
  EXPECT_EQUAL(params.floatParameter.value(), 1.5f);
  EXPECT_EQUAL(params.intParameter, 2);
  EXPECT(params.boolParameter);
  EXPECT(params.optFloatParameter.value() == boost::none);
  EXPECT(params.optDateTimeParameter.value() == boost::none);
  EXPECT(params.optDurationParameter.value() == boost::none);
  EXPECT(params.optPartialDateTimeParameter.value() == boost::none);
  EXPECT_EQUAL(params.reqFloatParameter, 3.0f);
  EXPECT_EQUAL(params.reqFloatParameter.value(), 3.0f);
  EXPECT_EQUAL(params.reqDurationParameter.value(), util::Duration("PT1H"));
  EXPECT(params.fruitParameter == Fruit::ORANGE);
  EXPECT(params.rangeParameter.value().minParameter == 0.0f);
  EXPECT(params.rangeParameter.value().maxParameter == 0.0f);
  EXPECT(params.intParameters.value().empty());
  EXPECT(params.rangeParameters.value().empty());
  EXPECT(params.variablesParameter.value() == oops::Variables());
  EXPECT(params.setIntParameter.value() == std::set<int>());
  EXPECT(params.anyOfParameter.value().as<std::vector<int>>() == std::vector<int>({10, 20}));
  EXPECT(params.optAnyOfParameter.value() == boost::none);
  EXPECT_NOT(params.optNullParameter.value());
  EXPECT(params.embeddedParameters.intParameter.value() == 3);
  EXPECT(params.embeddedParameters.optDateTimeParameter.value() == boost::none);
}

void testCorrectValues() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "full");
  EXPECT_NO_THROW(params.validate(fullConf));
  params.deserialize(fullConf);

  EXPECT_EQUAL(params.floatParameter, 3.5f);
  EXPECT_EQUAL(params.floatParameter.value(), 3.5f);
  EXPECT_EQUAL(params.intParameter, 4);
  EXPECT(!params.boolParameter);
  EXPECT(params.optFloatParameter.value() != boost::none);
  EXPECT_EQUAL(params.optFloatParameter.value().get(), 5.5f);
  EXPECT(params.optDateTimeParameter.value() != boost::none);
  EXPECT_EQUAL(params.optDateTimeParameter.value().get(), util::DateTime(2010, 2, 3, 4, 5, 6));
  EXPECT(params.optDurationParameter.value() != boost::none);
  EXPECT_EQUAL(params.optDurationParameter.value().get(), util::Duration("PT01H02M03S"));
  EXPECT(params.optPartialDateTimeParameter.value() != boost::none);
  EXPECT(params.optPartialDateTimeParameter.value().get() ==
         util::PartialDateTime(2010, -1, 3, 4, 5, 6));
  EXPECT_EQUAL(params.reqFloatParameter, 6.0f);
  EXPECT_EQUAL(params.reqFloatParameter.value(), 6.0f);
  EXPECT_EQUAL(params.reqDurationParameter.value(), util::Duration("PT06H30M"));
  EXPECT(params.fruitParameter == Fruit::APPLE);
  EXPECT(params.rangeParameter.value().minParameter == 7.0f);
  EXPECT(params.rangeParameter.value().maxParameter == 8.5f);
  EXPECT(params.intParameters.value() == std::vector<int>({1, 2}));
  EXPECT(params.rangeParameters.value().size() == 2);
  EXPECT(params.rangeParameters.value()[0].minParameter == 9.0f);
  EXPECT(params.rangeParameters.value()[0].maxParameter == 10.0f);
  EXPECT(params.rangeParameters.value()[1].minParameter == 11.0f);
  EXPECT(params.rangeParameters.value()[1].maxParameter == 12.0f);
  EXPECT(params.variablesParameter.value() ==
         oops::Variables({"u", "v"}, std::vector<int>({5, 6, 7})));
  EXPECT(params.setIntParameter.value() == std::set<int>({2, 4, 5, 6, 8}));
  EXPECT(params.anyOfParameter.value().as<std::string>() == "dog");
  EXPECT(params.optAnyOfParameter.value() != boost::none);
  EXPECT_EQUAL(params.optAnyOfParameter.value()->as<std::vector<int>>(),
               std::vector<int>({1, 2, 3, 4}));
  EXPECT(params.optNullParameter.value());
  EXPECT(params.embeddedParameters.intParameter.value() == 13);
  EXPECT(params.embeddedParameters.optDateTimeParameter.value() != boost::none);
  EXPECT_EQUAL(params.embeddedParameters.optDateTimeParameter.value().get(),
               util::DateTime(2010, 3, 4, 5, 6, 7));
}

void misspelledParameterNames() {
  MyOptionalParameters floatParam;
  const::eckit::LocalConfiguration floatConf(TestEnvironment::config(), "misspelled_float");
  if (validationSupported)
    EXPECT_THROWS_MSG(floatParam.validate(floatConf), "additional properties are not allowed");

  MyOptionalParameters intParam;
  const eckit::LocalConfiguration intConf(TestEnvironment::config(), "misspelled_int");
  if (validationSupported)
    EXPECT_THROWS_MSG(intParam.validate(intConf), "additional properties are not allowed");

  MyOptionalParameters boolParam;
  const eckit::LocalConfiguration boolConf(TestEnvironment::config(), "misspelled_bool");
  if (validationSupported)
    EXPECT_THROWS_MSG(boolParam.validate(boolConf), "additional properties are not allowed");

  MyOptionalParameters dtParam;
  const eckit::LocalConfiguration dtConf(TestEnvironment::config(), "misspelled_dt");
  if (validationSupported)
    EXPECT_THROWS_MSG(dtParam.validate(dtConf), "additional properties are not allowed");

  MyOptionalParameters durParam;
  const eckit::LocalConfiguration durConf(TestEnvironment::config(), "misspelled_dur");
  if (validationSupported)
    EXPECT_THROWS_MSG(durParam.validate(durConf), "additional properties are not allowed");

  MyOptionalParameters fruitParam;
  const eckit::LocalConfiguration fruitConf(TestEnvironment::config(), "misspelled_fruit");
  if (validationSupported)
    EXPECT_THROWS_MSG(fruitParam.validate(fruitConf), "additional properties are not allowed");

  MyOptionalParameters intsParam;
  const eckit::LocalConfiguration intsConf(TestEnvironment::config(), "misspelled_ints");
  if (validationSupported)
    EXPECT_THROWS_MSG(intsParam.validate(intsConf), "additional properties are not allowed");

  MyOptionalParameters nestedParam;
  const eckit::LocalConfiguration nestedParamConf
          (TestEnvironment::config(), "misspelled_nested_param");
  if (validationSupported)
    EXPECT_THROWS_MSG(nestedParam.validate(nestedParamConf),
                      "additional properties are not allowed");

  MyOptionalParameters nestedParams;
  const eckit::LocalConfiguration nestedParamsConf
          (TestEnvironment::config(), "misspelled_nested_params");
  if (validationSupported)
    EXPECT_THROWS_MSG(nestedParams.validate(nestedParamsConf),
                      "additional properties are not allowed");

  MyOptionalParameters nestingParam;
  const eckit::LocalConfiguration nestingParamConf
          (TestEnvironment::config(), "misspelled_nesting_param");
  if (validationSupported)
    EXPECT_THROWS_MSG(nestingParam.validate(nestingParamConf),
                      "additional properties are not allowed");
}

void testSerialization() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "full");
  doTestSerialization<MyOptionalAndRequiredParameters>(fullConf);
}

void testToConfiguration() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration inputConfig(TestEnvironment::config(), "full");
  EXPECT_NO_THROW(params.validate(inputConfig));
  params.deserialize(inputConfig);

  eckit::LocalConfiguration serializeOutput;
  params.serialize(serializeOutput);

  eckit::LocalConfiguration toConfigurationOutput = params.toConfiguration();

  std::stringstream expectedStream;
  expectedStream << serializeOutput;
  const std::string expected = expectedStream.str();

  std::stringstream receivedStream;
  receivedStream << toConfigurationOutput;
  std::string received = receivedStream.str();

  EXPECT_EQUAL(received, expected);
}

void testIncorrectValueOfFloatParameter() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "error_in_float_parameter");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "unexpected value type");
  EXPECT_THROWS_AS(params.deserialize(conf), eckit::BadParameter);
}

void testIncorrectValueOfOptionalFloatParameter() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "error_in_opt_float_parameter");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "unexpected value type");
  EXPECT_THROWS_AS(params.deserialize(conf), eckit::BadParameter);
}


void testIncorrectValueOfOptionalPartialDateTimeParameter() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "error_in_opt_partialDT_parameter");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "ABCDEF is not a partial-date-time string");
  EXPECT_THROWS_AS(params.deserialize(conf), eckit::BadParameter);
}


void testIncorrectValueOfOptionalDateTimeParameter() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "error_in_opt_date_time_parameter");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "ABCDEF is not a date-time string");
  // Conversion from string to DateTime calls abort() on failure,
  // so we can't test this call. Leaving it commented-out in case this changes in future.
  // EXPECT_THROWS_AS(params.deserialize(conf), eckit::BadParameter);
}

void testIncorrectValueOfOptionalDurationParameter() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "error_in_opt_duration_parameter");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "ABCDEF is not a duration string");
  // Conversion from string to Duration calls abort() on failure,
  // so we can't test this call. Leaving it commented-out in case this changes in future.
  // EXPECT_THROWS_AS(params.deserialize(conf), eckit::BadParameter);
}

void testIncorrectValueOfEnumParameter() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "error_in_fruit_parameter");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "unrecognized enum value");
  EXPECT_THROWS_AS(params.deserialize(conf), eckit::BadParameter);
}

void testIncorrectValueOfIntParameters() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "error_in_int_parameters");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "unexpected value type");
  EXPECT_THROWS_AS(params.deserialize(conf), eckit::Exception);
}

void testIncorrectValueOfRangeParameters() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "error_in_range_parameters");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "unexpected value type");
  EXPECT_THROWS_AS(params.deserialize(conf), eckit::Exception);
}

void testIncorrectValueOfAnyOfParameter() {
  {
    // This YAML section sets any_of_parameter to a value that is neither a string nor a list of
    // ints. This should be detected at the validation stage.
    MyOptionalParameters params;
    const eckit::LocalConfiguration conf(TestEnvironment::config(), "error_in_any_of_parameter");
    if (validationSupported)
      EXPECT_THROWS_MSG(params.validate(conf), "no subschema has succeeded");
  }

  {
    // This YAML section sets any_of_parameter to a string, but the code attempts to load
    // it as a vector of ints. This can be detected only in the call to the as() method.
    MyOptionalAndRequiredParameters params;
    const eckit::LocalConfiguration conf(TestEnvironment::config(), "full");
    params.validate(conf);
    params.deserialize(conf);
    EXPECT_THROWS_MSG(params.anyOfParameter.value().as<std::vector<int>>(),
                      "unexpected value type");
  }

  {
    // This YAML section sets any_of_parameter to a vector of ints, but the code attempts to load
    // it as a string. This can be detected only in the call to the as() method.
    MyOptionalAndRequiredParameters params;
    const eckit::LocalConfiguration conf(TestEnvironment::config(), "alternative");
    params.validate(conf);
    params.deserialize(conf);
    EXPECT_THROWS_MSG(params.anyOfParameter.value().as<std::string>(),
                      "unexpected value type");
  }
}

void testMissingRequiredFloatParameter() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "missing_req_float_parameter");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf),
                      "required property 'req_float_parameter' not found in object");
  EXPECT_THROWS_AS(params.deserialize(conf), eckit::BadParameter);
}

void testMissingRequiredDurationParameter() {
  MyOptionalAndRequiredParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "missing_req_duration_parameter");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf),
                      "required property 'req_duration_parameter' not found in object");
  EXPECT_THROWS_AS(params.deserialize(conf), eckit::BadParameter);
}

// Tests of parameters storing maps and ScalarOrMaps

void testMapParameters(const MyMapParameters &params) {
  // Map parameters
  EXPECT_EQUAL(params.intToFloatMapParameter.value().at(5),
               1.5f);
  EXPECT_EQUAL(params.intToFloatMapParameter.value().at(7),
               3.0f);
  EXPECT(params.intToFloatMapParameter.value().find(123456) ==
         params.intToFloatMapParameter.value().end());

  EXPECT_EQUAL(params.stringToDurationMapParameter.value().at("day"),
               util::Duration("PT16H"));
  EXPECT_EQUAL(params.stringToDurationMapParameter.value().at("night"),
               util::Duration("PT8H"));
  EXPECT(params.stringToDurationMapParameter.value().find("abcdef") ==
         params.stringToDurationMapParameter.value().end());

  // Scalar-or-map parameters set to maps
  EXPECT_NOT(params.floatOrIntToFloatMapParameter1.value().isScalar());
  EXPECT_EQUAL(params.floatOrIntToFloatMapParameter1.value().at(6),
               2.5f);
  EXPECT_EQUAL(params.floatOrIntToFloatMapParameter1.value().at(8),
               4.0f);
  EXPECT_NOT(params.floatOrIntToFloatMapParameter1.value().contains(1));

  EXPECT_NOT(params.durationOrStringToDurationMapParameter1.value().isScalar());
  EXPECT_EQUAL(params.durationOrStringToDurationMapParameter1.value().at("day"),
               util::Duration("PT14H"));
  EXPECT_EQUAL(params.durationOrStringToDurationMapParameter1.value().at("night"),
               util::Duration("PT10H"));
  EXPECT_NOT(params.durationOrStringToDurationMapParameter1.value().contains("abcdef"));

  // Scalar-or-map parameters set to scalars
  EXPECT(params.floatOrIntToFloatMapParameter2.value().isScalar());
  EXPECT_EQUAL(params.floatOrIntToFloatMapParameter2.value().at(6),
               3.5f);
  EXPECT_EQUAL(params.floatOrIntToFloatMapParameter2.value().at(8),
               3.5f);
  EXPECT(params.floatOrIntToFloatMapParameter2.value().contains(123456));
  EXPECT_EQUAL(params.floatOrIntToFloatMapParameter2.value().at(123456),
               3.5f);

  EXPECT(params.durationOrStringToDurationMapParameter2.value().isScalar());
  EXPECT_EQUAL(params.durationOrStringToDurationMapParameter2.value().at("day"),
               util::Duration("PT12H"));
  EXPECT_EQUAL(params.durationOrStringToDurationMapParameter2.value().at("night"),
               util::Duration("PT12H"));
  EXPECT(params.durationOrStringToDurationMapParameter2.value().contains("abcdef"));
  EXPECT_EQUAL(params.durationOrStringToDurationMapParameter2.value().at("abcdef"),
               util::Duration("PT12H"));
}

void testMapParametersYamlStyleQuotedKeys() {
  MyMapParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "map_parameter_yaml_style_quoted_keys");
  EXPECT_NO_THROW(params.validate(conf));
  params.deserialize(conf);
  testMapParameters(params);
}

void testMapParametersYamlStyleUnquotedKeys() {
  MyMapParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "map_parameter_yaml_style_unquoted_keys");
  EXPECT_NO_THROW(params.validate(conf));
  params.deserialize(conf);
  testMapParameters(params);
}

void testMapParametersJsonStyleQuotedKeys() {
  MyMapParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "map_parameter_json_style_quoted_keys");
  EXPECT_NO_THROW(params.validate(conf));
  params.deserialize(conf);
  testMapParameters(params);
}

void testMapParametersJsonStyleUnquotedKeys() {
  MyMapParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "map_parameter_json_style_unquoted_keys");
  EXPECT_NO_THROW(params.validate(conf));
  params.deserialize(conf);
  testMapParameters(params);
}

void testMapParametersSerialization() {
  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "map_parameter_json_style_quoted_keys");
  doTestSerialization<MyMapParameters>(conf);
}

// Parameters storing std::set<int> objects

void testSetIntParameters() {
  {
    MyOptionalParameters params;
    const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                         "set_int_single_number");
    EXPECT_NO_THROW(params.validate(conf));
    params.deserialize(conf);
    EXPECT(params.setIntParameter.value() == std::set<int>({5}));
  }

  {
    MyOptionalParameters params;
    const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                         "set_int_range");
    EXPECT_NO_THROW(params.validate(conf));
    params.deserialize(conf);
    EXPECT(params.setIntParameter.value() == std::set<int>({3, 4, 5}));
  }

  {
    MyOptionalParameters params;
    const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                         "set_int_multiple_numbers_and_ranges");
    EXPECT_NO_THROW(params.validate(conf));
    params.deserialize(conf);
    EXPECT(params.setIntParameter.value() == std::set<int>({3, 4, 5, 7, 10, 11, 13}));
  }

  {
    MyOptionalParameters params;
    const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                         "set_int_invalid_range");
    // Note: this syntax error won't be detected at the validation stage, but only at the
    // deserialization stage
    EXPECT_THROWS_MSG(params.deserialize(conf),
                      "isn't a list of comma-separated integers or ranges of integers");
  }
}

// Parameters storing Variables objects

void testVariablesDeserializationWithoutChannels() {
  VariablesParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(), "variables_without_channels");
  EXPECT_NO_THROW(params.validate(conf));
  params.deserialize(conf);

  const oops::Variables expectedFilterVariables(conf, "filter_variables");
  const oops::Variables expectedOperatorVariables(conf, "operator_variables");

  EXPECT_EQUAL(params.filterVariables.value(), expectedFilterVariables);
  EXPECT_EQUAL(params.operatorVariables.value(), expectedOperatorVariables);
}

void testVariablesDeserializationWithChannels() {
  VariablesParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(), "variables_with_channels");
  EXPECT_NO_THROW(params.validate(conf));
  params.deserialize(conf);

  const oops::Variables expectedFilterVariables(conf, "filter_variables");
  const oops::Variables expectedOperatorVariables(conf, "operator_variables");

  EXPECT_EQUAL(params.filterVariables.value(), expectedFilterVariables);
  EXPECT_EQUAL(params.operatorVariables.value(), expectedOperatorVariables);
}

void testVariablesSerializationWithoutChannels() {
  VariablesParameters params;
  const eckit::LocalConfiguration conf(TestEnvironment::config(), "variables_without_channels");
  doTestSerialization<VariablesParameters>(conf);
}

void testVariablesSerializationWithChannels() {
  const eckit::LocalConfiguration conf(TestEnvironment::config(), "variables_with_channels");
  doTestSerialization<VariablesParameters>(conf);
}

void testCompositeVariablesSerialization() {
  // Variable objects containing variables that don't share the same channel suffixes
  // cannot be represented by a single Configuration object.
  {
    oops::Variables var1({"air_temperature", "air_pressure"}, std::vector<int>({5, 6, 7}));
    oops::Variables var2({"relative_humidity"}, std::vector<int>({1, 2, 3}));
    var1 += var2;

    eckit::LocalConfiguration conf;
    EXPECT_THROWS(oops::ParameterTraits<oops::Variables>::set(conf, "name", var1));
  }

  // Case with some variables having channel suffixes and others not
  {
    oops::Variables var1({"air_temperature", "air_pressure"}, std::vector<int>({5, 6, 7}));
    // var2 won't have channels
    const eckit::LocalConfiguration helperConf(TestEnvironment::config(),
                                               "variables_without_channels");
    oops::Variables var2(helperConf, "operator_variables");
    var1 += var2;

    eckit::LocalConfiguration conf;
    EXPECT_THROWS(oops::ParameterTraits<oops::Variables>::set(conf, "name", var1));
  }

  // Case with all variables having the same channel suffixes
  {
    oops::Variables var1({"air_temperature", "air_pressure"}, std::vector<int>({5, 6, 7}));
    oops::Variables var2({"relative_humidity"}, std::vector<int>({5, 6, 7}));
    var1 += var2;

    eckit::LocalConfiguration conf;
    EXPECT_NO_THROW(oops::ParameterTraits<oops::Variables>::set(conf, "name", var1));
  }
}

// Tests of special member functions

void expectMatchesFullConf(const MyOptionalAndRequiredParameters &params) {
  EXPECT_EQUAL(params.floatParameter, 3.5f);
  EXPECT_EQUAL(params.rangeParameter.value().minParameter, 7.0f);
  EXPECT_EQUAL(params.rangeParameters.value()[0].minParameter, 9.0f);
  EXPECT_EQUAL(params.setIntParameter.value(), std::set<int>({2, 4, 5, 6, 8}));
  EXPECT(params.optNullParameter.value());
  EXPECT(params.optAnyOfParameter.value() != boost::none);
  EXPECT_EQUAL(params.optAnyOfParameter.value()->as<std::vector<int>>(),
               std::vector<int>({1, 2, 3, 4}));
  EXPECT(params.embeddedParameters.intParameter.value() == 13);
}

void expectMatchesAlternativeConf(const MyOptionalAndRequiredParameters &params) {
  EXPECT_EQUAL(params.floatParameter, 13.5f);
  EXPECT_EQUAL(params.rangeParameter.value().minParameter, 17.0f);
  EXPECT_EQUAL(params.rangeParameters.value()[0].minParameter, 19.0f);
  EXPECT_EQUAL(params.setIntParameter.value(), std::set<int>({3}));
  EXPECT(params.optNullParameter.value());
  EXPECT(params.optAnyOfParameter.value() != boost::none);
  EXPECT_EQUAL(params.optAnyOfParameter.value()->as<std::string>(), "cat");
  EXPECT(params.embeddedParameters.intParameter.value() == 23);
}

void testCopyConstructor() {
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "full");
  const eckit::LocalConfiguration alternativeConf(TestEnvironment::config(), "alternative");

  MyOptionalAndRequiredParameters params;
  EXPECT_NO_THROW(params.validate(fullConf));
  params.deserialize(fullConf);

  MyOptionalAndRequiredParameters otherParams = params;
  expectMatchesFullConf(otherParams);

  EXPECT_NO_THROW(params.validate(alternativeConf));
  otherParams.deserialize(alternativeConf);
  expectMatchesAlternativeConf(otherParams);

  expectMatchesFullConf(params);

  params.deserialize(alternativeConf);
  expectMatchesAlternativeConf(params);
}

void testMoveConstructor() {
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "full");
  const eckit::LocalConfiguration alternativeConf(TestEnvironment::config(), "alternative");

  MyOptionalAndRequiredParameters params;
  params.deserialize(fullConf);

  MyOptionalAndRequiredParameters otherParams = std::move(params);
  expectMatchesFullConf(otherParams);

  otherParams.deserialize(alternativeConf);
  expectMatchesAlternativeConf(otherParams);
}

void testCopyAssignmentOperator() {
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "full");
  const eckit::LocalConfiguration alternativeConf(TestEnvironment::config(), "alternative");

  MyOptionalAndRequiredParameters params;
  params.deserialize(fullConf);

  MyOptionalAndRequiredParameters otherParams;
  otherParams = params;
  expectMatchesFullConf(otherParams);

  otherParams.deserialize(alternativeConf);
  expectMatchesAlternativeConf(otherParams);

  expectMatchesFullConf(params);

  params.deserialize(alternativeConf);
  expectMatchesAlternativeConf(params);
}

void testMoveAssignmentOperator() {
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "full");
  const eckit::LocalConfiguration alternativeConf(TestEnvironment::config(), "alternative");

  MyOptionalAndRequiredParameters params;
  params.deserialize(fullConf);

  MyOptionalAndRequiredParameters otherParams;
  otherParams = std::move(params);
  expectMatchesFullConf(otherParams);

  otherParams.deserialize(alternativeConf);
  expectMatchesAlternativeConf(otherParams);
}

// Tests of polymorphic parameters

void testPolymorphicParametersDeserialization() {
  AllDeviceParameters params;
  EXPECT(params.optionalDevice.value().device.value() == nullptr);
  EXPECT_EQUAL(params.deviceWithDefault.value().device.id(), "screen");

  const eckit::LocalConfiguration conf(TestEnvironment::config(), "device");
  EXPECT_NO_THROW(params.validate(conf));
  params.deserialize(conf);

  {
    EXPECT(params.optionalDevice.value().device.value() != nullptr);
    EXPECT(params.optionalDevice.value().device.id() != boost::none);
    EXPECT_EQUAL(*params.optionalDevice.value().device.id(), "printer");
    const DeviceTypeDependentParameters &device =
        *params.optionalDevice.value().device.value();
    auto deviceParameters = dynamic_cast<const PrinterParameters*>(&device);
    EXPECT(deviceParameters != nullptr);
    EXPECT_EQUAL(deviceParameters->paperFormat.value(), "Letter");
  }
  {
    EXPECT_EQUAL(params.requiredDevice.value().device.id(), "screen");
    const DeviceTypeDependentParameters &device = params.requiredDevice.value().device.value();
    auto deviceParameters = dynamic_cast<const ScreenParameters*>(&device);
    EXPECT(deviceParameters != nullptr);
    EXPECT(deviceParameters->diameter.value() != boost::none);
    EXPECT_EQUAL(*deviceParameters->diameter.value(), 30.0f);
  }
  {
    EXPECT_EQUAL(params.deviceWithDefault.value().device.id(), "printer");
    const DeviceTypeDependentParameters &device =
        params.deviceWithDefault.value().device.value();
    auto deviceParameters = dynamic_cast<const PrinterParameters*>(&device);
    EXPECT(deviceParameters != nullptr);
    EXPECT_EQUAL(deviceParameters->paperFormat.value(), "A3");
  }
}

// Tests behavior of PolymorphicParameter<PARAMETERS, ...> when the YAML file doesn't
// contain the keyword used to select the type of the subclass of PARAMETERS to be instantiated.
void testPolymorphicParametersIncompleteDeserialization() {
  AllDeviceParameters params;

  const eckit::LocalConfiguration conf(TestEnvironment::config(), "device_with_default_not_set");
  EXPECT_NO_THROW(params.validate(conf));
  params.deserialize(conf);

  {
    EXPECT_EQUAL(params.deviceWithDefault.value().device.id(), "screen");
    const DeviceTypeDependentParameters &device =
        params.deviceWithDefault.value().device.value();
    auto deviceParameters = dynamic_cast<const ScreenParameters*>(&device);
    EXPECT(deviceParameters != nullptr);
    EXPECT(deviceParameters->diameter.value() == boost::none);
  }
}

// Tests behavior of OptionalPolymorphicParameter<PARAMETERS, ...> when the YAML file
// doesn't contain the keyword used to select the type of the subclass of PARAMETERS to be
// instantiated.
void testOptionalPolymorphicParametersIncompleteDeserialization() {
  AllDeviceParameters params;

  const eckit::LocalConfiguration conf(TestEnvironment::config(), "optional_device_not_set");
  EXPECT_NO_THROW(params.validate(conf));
  params.deserialize(conf);

  {
    EXPECT(params.optionalDevice.value().device.id() == boost::none);
    EXPECT(params.optionalDevice.value().device.value() == nullptr);
  }
}

// Tests behavior of RequiredPolymorphicParameter<PARAMETERS, ...> when the YAML file
// doesn't contain the keyword used to select the type of the subclass of PARAMETERS to be
// instantiated.
void testRequiredPolymorphicParametersIncompleteDeserialization() {
  AllDeviceParameters params;

  const eckit::LocalConfiguration conf(TestEnvironment::config(), "required_device_not_set");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "required property 'type' not found in object");
  EXPECT_THROWS(params.deserialize(conf));
}

// Tests behavior of PolymorphicParameter<PARAMETERS, ...> when the value of the key
// used to select the type of the subclass of PARAMETERS does not match any registered type.
void testInvalidPolymorphicParametersId() {
  AllDeviceParameters params;

  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "invalid_type_of_device_with_default");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "unrecognized enum value");
  EXPECT_THROWS(params.deserialize(conf));
}

// Tests behavior of OptionalPolymorphicParameter<PARAMETERS, ...> when the value of the key
// used to select the type of the subclass of PARAMETERS does not match any registered type.
void testInvalidOptionalPolymorphicParametersId() {
  AllDeviceParameters params;

  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "invalid_type_of_optional_device");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "unrecognized enum value");
  EXPECT_THROWS(params.deserialize(conf));
}

// Tests behavior of RequiredPolymorphicParameter<PARAMETERS, ...> when the value of the key
// used to select the type of the subclass of PARAMETERS does not match any registered type.
void testInvalidRequiredPolymorphicParametersId() {
  AllDeviceParameters params;

  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "invalid_type_of_required_device");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "unrecognized enum value");
  EXPECT_THROWS(params.deserialize(conf));
}

void testMisspelledPropertyOfPolymorphicParameters() {
  AllDeviceParameters params;

  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "misspelled_diameter_of_device_with_default");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "additional properties are not allowed");
}

void testMisspelledPropertyOfOptionalPolymorphicParameters() {
  AllDeviceParameters params;

  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "misspelled_diameter_of_optional_device");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "additional properties are not allowed");
}

void testMisspelledPropertyOfRequiredPolymorphicParameters() {
  AllDeviceParameters params;

  const eckit::LocalConfiguration conf(TestEnvironment::config(),
                                       "misspelled_diameter_of_required_device");
  if (validationSupported)
    EXPECT_THROWS_MSG(params.validate(conf), "additional properties are not allowed");
}

void expectMatchesFullConf(const AllDeviceParameters &params) {
  {
    const DeviceTypeDependentParameters *device = params.optionalDevice.value().device.value();
    EXPECT(device != nullptr);
    auto deviceParameters = dynamic_cast<const PrinterParameters*>(device);
    EXPECT(deviceParameters != nullptr);
    EXPECT_EQUAL(deviceParameters->paperFormat.value(), "Letter");
  }
  {
    const DeviceTypeDependentParameters &device = params.requiredDevice.value().device.value();
    auto deviceParameters = dynamic_cast<const ScreenParameters*>(&device);
    EXPECT(deviceParameters != nullptr);
    EXPECT(deviceParameters->diameter.value() != boost::none);
    EXPECT_EQUAL(*deviceParameters->diameter.value(), 30.0f);
  }
  {
    const DeviceTypeDependentParameters &device = params.deviceWithDefault.value().device.value();
    auto deviceParameters = dynamic_cast<const PrinterParameters*>(&device);
    EXPECT(deviceParameters != nullptr);
    EXPECT_EQUAL(deviceParameters->paperFormat.value(), "A3");
  }
}

void expectMatchesAlternativeConf(const AllDeviceParameters &params) {
  {
    const DeviceTypeDependentParameters *device = params.optionalDevice.value().device.value();
    EXPECT(device != nullptr);
    auto deviceParameters = dynamic_cast<const ScreenParameters*>(device);
    EXPECT(deviceParameters != nullptr);
    EXPECT(deviceParameters->diameter.value() != boost::none);
    EXPECT_EQUAL(*deviceParameters->diameter.value(), 40.0f);
  }
  {
    const DeviceTypeDependentParameters &device = params.requiredDevice.value().device.value();
    auto deviceParameters = dynamic_cast<const PrinterParameters*>(&device);
    EXPECT(deviceParameters != nullptr);
    EXPECT_EQUAL(deviceParameters->paperFormat.value(), "A5");
  }
  {
    const DeviceTypeDependentParameters &device = params.deviceWithDefault.value().device.value();
    auto deviceParameters = dynamic_cast<const ScreenParameters*>(&device);
    EXPECT(deviceParameters != nullptr);
    EXPECT(deviceParameters->diameter.value() != boost::none);
    EXPECT_EQUAL(*deviceParameters->diameter.value(), 20.0f);
  }
}

void testPolymorphicParametersCopyConstructor() {
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "device");
  const eckit::LocalConfiguration alternativeConf(TestEnvironment::config(), "alternative_device");

  AllDeviceParameters params;
  EXPECT_NO_THROW(params.validate(fullConf));
  params.deserialize(fullConf);

  AllDeviceParameters otherParams = params;
  expectMatchesFullConf(otherParams);

  EXPECT_NO_THROW(params.validate(alternativeConf));
  otherParams.deserialize(alternativeConf);
  expectMatchesAlternativeConf(otherParams);

  expectMatchesFullConf(params);

  params.deserialize(alternativeConf);
  expectMatchesAlternativeConf(params);
}

void testPolymorphicParametersMoveConstructor() {
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "device");
  const eckit::LocalConfiguration alternativeConf(TestEnvironment::config(), "alternative_device");

  AllDeviceParameters params;
  params.deserialize(fullConf);

  AllDeviceParameters otherParams = std::move(params);
  expectMatchesFullConf(otherParams);

  otherParams.deserialize(alternativeConf);
  expectMatchesAlternativeConf(otherParams);
}

void testPolymorphicParametersCopyAssignmentOperator() {
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "device");
  const eckit::LocalConfiguration alternativeConf(TestEnvironment::config(), "alternative_device");

  AllDeviceParameters params;
  params.deserialize(fullConf);

  AllDeviceParameters otherParams;
  otherParams = params;
  expectMatchesFullConf(otherParams);

  otherParams.deserialize(alternativeConf);
  expectMatchesAlternativeConf(otherParams);

  expectMatchesFullConf(params);

  params.deserialize(alternativeConf);
  expectMatchesAlternativeConf(params);
}

void testPolymorphicParametersMoveAssignmentOperator() {
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "device");
  const eckit::LocalConfiguration alternativeConf(TestEnvironment::config(), "alternative_device");

  AllDeviceParameters params;
  params.deserialize(fullConf);

  AllDeviceParameters otherParams;
  otherParams = std::move(params);
  expectMatchesFullConf(otherParams);

  otherParams.deserialize(alternativeConf);
  expectMatchesAlternativeConf(otherParams);
}

void testPolymorphicParametersSerialization() {
  const eckit::LocalConfiguration conf(TestEnvironment::config(), "device");
  doTestSerialization<AllDeviceParameters>(conf);
}

// Constraint tests

// - Generic routines

template <typename ParametersType>
void doTestMinConstraint() {
  const eckit::LocalConfiguration conf(TestEnvironment::config());
  {
    const eckit::LocalConfiguration validConf(conf, "constraints_met");
    ParametersType params;
    EXPECT_NO_THROW(params.validate(validConf));
    EXPECT_NO_THROW(params.deserialize(validConf));
  }
  {
    const eckit::LocalConfiguration invalidConf(conf, "int_min_constraint_not_met");
    ParametersType params;
    if (validationSupported)
      EXPECT_THROWS_MSG(params.validate(invalidConf), "value is below minimum");
    EXPECT_THROWS(params.deserialize(invalidConf));
  }
  {
    const eckit::LocalConfiguration invalidConf(conf, "float_min_constraint_not_met");
    ParametersType params;
    if (validationSupported)
      EXPECT_THROWS_MSG(params.validate(invalidConf), "value is below minimum");
    EXPECT_THROWS(params.deserialize(invalidConf));
  }
}

template <typename ParametersType>
void doTestExclusiveMinConstraint() {
  const eckit::LocalConfiguration conf(TestEnvironment::config());
  {
    const eckit::LocalConfiguration validConf(conf, "constraints_met");
    ParametersType params;
    EXPECT_NO_THROW(params.validate(validConf));
    EXPECT_NO_THROW(params.deserialize(validConf));
  }
  {
    const eckit::LocalConfiguration invalidConf(conf, "int_exclusive_min_constraint_not_met");
    ParametersType params;
    if (validationSupported)
      EXPECT_THROWS(params.validate(invalidConf));
    EXPECT_THROWS(params.deserialize(invalidConf));
  }
  {
    const eckit::LocalConfiguration invalidConf(conf, "float_exclusive_min_constraint_not_met");
    ParametersType params;
    if (validationSupported)
      EXPECT_THROWS(params.validate(invalidConf));
    EXPECT_THROWS(params.deserialize(invalidConf));
  }
}

template <typename ParametersType>
void doTestMaxConstraint() {
  const eckit::LocalConfiguration conf(TestEnvironment::config());
  {
    const eckit::LocalConfiguration validConf(conf, "constraints_met");
    ParametersType params;
    EXPECT_NO_THROW(params.validate(validConf));
    EXPECT_NO_THROW(params.deserialize(validConf));
  }
  {
    const eckit::LocalConfiguration invalidConf(conf, "int_max_constraint_not_met");
    ParametersType params;
    if (validationSupported)
      EXPECT_THROWS_MSG(params.validate(invalidConf), "value exceeds maximum");
    EXPECT_THROWS(params.deserialize(invalidConf));
  }
  {
    const eckit::LocalConfiguration invalidConf(conf, "float_max_constraint_not_met");
    ParametersType params;
    if (validationSupported)
      EXPECT_THROWS_MSG(params.validate(invalidConf), "value exceeds maximum");
    EXPECT_THROWS(params.deserialize(invalidConf));
  }
}

template <typename ParametersType>
void doTestExclusiveMaxConstraint() {
  const eckit::LocalConfiguration conf(TestEnvironment::config());
  {
    const eckit::LocalConfiguration validConf(conf, "constraints_met");
    ParametersType params;
    EXPECT_NO_THROW(params.validate(validConf));
    EXPECT_NO_THROW(params.deserialize(validConf));
  }
  {
    const eckit::LocalConfiguration invalidConf(conf, "int_exclusive_max_constraint_not_met");
    ParametersType params;
    if (validationSupported)
      EXPECT_THROWS(params.validate(invalidConf));
    EXPECT_THROWS(params.deserialize(invalidConf));
  }
  {
    const eckit::LocalConfiguration invalidConf(conf, "float_exclusive_max_constraint_not_met");
    ParametersType params;
    if (validationSupported)
      EXPECT_THROWS(params.validate(invalidConf));
    EXPECT_THROWS(params.deserialize(invalidConf));
  }
}

// - Minimum constraints

void testParametersWithMinConstraint() {
  doTestMinConstraint<ConstrainedParameters>();
}

void testRequiredParametersWithMinConstraint() {
  doTestMinConstraint<ConstrainedRequiredParameters>();
}

void testOptionalParametersWithMinConstraint() {
  doTestMinConstraint<ConstrainedOptionalParameters>();
}

// - Exclusive minimum constraints

void testParametersWithExclusiveMinConstraint() {
  doTestExclusiveMinConstraint<ConstrainedParameters>();
}

void testRequiredParametersWithExclusiveMinConstraint() {
  doTestExclusiveMinConstraint<ConstrainedRequiredParameters>();
}

void testOptionalParametersWithExclusiveMinConstraint() {
  doTestExclusiveMinConstraint<ConstrainedOptionalParameters>();
}

// - Maximum constraints

void testParametersWithMaxConstraint() {
  doTestMaxConstraint<ConstrainedParameters>();
}

void testRequiredParametersWithMaxConstraint() {
  doTestMaxConstraint<ConstrainedRequiredParameters>();
}

void testOptionalParametersWithMaxConstraint() {
  doTestMaxConstraint<ConstrainedOptionalParameters>();
}

// - Exclusive maximum constraints

void testParametersWithExclusiveMaxConstraint() {
  doTestExclusiveMaxConstraint<ConstrainedParameters>();
}

void testRequiredParametersWithExclusiveMaxConstraint() {
  doTestExclusiveMaxConstraint<ConstrainedRequiredParameters>();
}

void testOptionalParametersWithExclusiveMaxConstraint() {
  doTestExclusiveMaxConstraint<ConstrainedOptionalParameters>();
}

// ConfigurationParameter

void testConfigurationParameter() {
  const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "full");
  IncompleteParameters params;
  EXPECT_NO_THROW(params.validate(fullConf));
  params.deserialize(fullConf);

  EXPECT_EQUAL(params.floatParameter, 3.5f);
  EXPECT_EQUAL(params.intParameter, 4);
  // params.config should contain values loaded also into other, more specific parameters...
  EXPECT_EQUAL(params.config.value().getFloat("float_parameter"), 3.5f);
  EXPECT_EQUAL(params.config.value().getInt("int_parameter"), 4);
  // as well as all other values, for example this one:
  EXPECT_NOT(params.config.value().getBool("bool_parameter"));
}

// Detection of schema conflicts

void testSchemaConflictDetection() {
  ConstrainedThenUnconstrainedParameters constrainedThenUnconstrainedParameters;
  EXPECT_NO_THROW(constrainedThenUnconstrainedParameters.jsonSchema());
  doTestMinConstraint<ConstrainedThenUnconstrainedParameters>();

  UnconstrainedThenConstrainedParameters unconstrainedThenConstrainedParameters;
  EXPECT_NO_THROW(unconstrainedThenConstrainedParameters.jsonSchema());
  doTestMinConstraint<UnconstrainedThenConstrainedParameters>();

  ConflictingParameters conflictingParameters;
  EXPECT_THROWS(conflictingParameters.jsonSchema());
}

// IgnoreOtherParameters

void testIgnoreOtherParameters() {
  {
    // We are tolerant: we ignore unregistred parameters...
    const eckit::LocalConfiguration fullConf(TestEnvironment::config(), "full");
    TolerantParameters params;
    EXPECT_NO_THROW(params.validate(fullConf));
    params.deserialize(fullConf);

    EXPECT_EQUAL(params.floatParameter, 3.5f);
    EXPECT_EQUAL(params.intParameter, 4);
  }

  {
    // ... but everything in moderation: we still detect errors in registered parameters
    const eckit::LocalConfiguration badConf(TestEnvironment::config(), "error_in_float_parameter");
    TolerantParameters params;
    if (validationSupported)
      EXPECT_THROWS(params.validate(badConf));
  }
}

// validateAndDeserialize

void testValidateAndDeserialize() {
  {
    const eckit::LocalConfiguration conf(TestEnvironment::config(), "full");
    EXPECT_NO_THROW(oops::validateAndDeserialize<MyOptionalAndRequiredParameters>(conf));
  }

  {
    const::eckit::LocalConfiguration conf(TestEnvironment::config(), "misspelled_float");
    if (validationSupported)
      EXPECT_THROWS(oops::validateAndDeserialize<MyOptionalParameters>(conf));
  }
}

// HasParameters_

void testHasParameters_() {
  EXPECT_NOT(oops::HasParameters_<WithoutParameters_>::value);
  EXPECT_NOT(oops::HasParameters_<WithParameters_NotDerivedFromOopsParameters>::value);
  EXPECT_NOT(oops::HasParameters_<WithParameters_DerivedPrivatelyFromOopsParameters>::value);
  EXPECT_NOT(oops::HasParameters_<WithParameters_DerivedFromNonOopsParameters>::value);
  EXPECT(oops::HasParameters_<WithParameters_DerivedFromOopsParameters>::value);
}


class Parameters : public oops::Test {
 private:
  std::string testid() const override {return "test::Parameters";}

  void register_tests() const override {
    std::vector<eckit::testing::Test>& ts = eckit::testing::specification();

    ts.emplace_back(CASE("util/Parameters/defaultValues") {
                      testDefaultValues();
                    });
    ts.emplace_back(CASE("util/Parameters/correctValues") {
                      testCorrectValues();
                    });
    ts.emplace_back(CASE("util/Parameters/misspelledParameterNames") {
                      misspelledParameterNames();
                    });
    ts.emplace_back(CASE("util/Parameters/incorrectValueOfFloatParameter") {
                      testIncorrectValueOfFloatParameter();
                    });
    ts.emplace_back(CASE("util/Parameters/incorrectValueOfOptionalFloatParameter") {
                      testIncorrectValueOfOptionalFloatParameter();
                    });
    ts.emplace_back(CASE("util/Parameters/incorrectValueOfOptionalPartialDateTimeParameter") {
                    testIncorrectValueOfOptionalPartialDateTimeParameter();
                    });
    ts.emplace_back(CASE("util/Parameters/incorrectValueOfOptionalDateTimeParameter") {
                      testIncorrectValueOfOptionalDateTimeParameter();
                    });
    ts.emplace_back(CASE("util/Parameters/incorrectValueOfOptionalDurationParameter") {
                      testIncorrectValueOfOptionalDurationParameter();
                    });
    ts.emplace_back(CASE("util/Parameters/incorrectValueOfEnumParameter") {
                      testIncorrectValueOfEnumParameter();
                    });
    ts.emplace_back(CASE("util/Parameters/testIncorrectValueOfIntParameters") {
                      testIncorrectValueOfIntParameters();
                    });
    ts.emplace_back(CASE("util/Parameters/testIncorrectValueOfRangeParameters") {
                      testIncorrectValueOfRangeParameters();
                    });
    ts.emplace_back(CASE("util/Parameters/testIncorrectValueOfAnyOfParameter") {
                      testIncorrectValueOfAnyOfParameter();
                    });
    ts.emplace_back(CASE("util/Parameters/testMissingRequiredFloatParameter") {
                      testMissingRequiredFloatParameter();
                    });
    ts.emplace_back(CASE("util/Parameters/testMissingRequiredDurationParameter") {
                      testMissingRequiredDurationParameter();
                    });

    ts.emplace_back(CASE("util/Parameters/testCopyConstructor") {
                      testCopyConstructor();
                    });
    ts.emplace_back(CASE("util/Parameters/testMoveConstructor") {
                      testMoveConstructor();
                    });
    ts.emplace_back(CASE("util/Parameters/testCopyAssignmentOperator") {
                      testCopyAssignmentOperator();
                    });
    ts.emplace_back(CASE("util/Parameters/testMoveAssignmentOperator") {
                      testMoveAssignmentOperator();
                    });
    ts.emplace_back(CASE("util/Parameters/serialization") {
                      testSerialization();
                    });
    ts.emplace_back(CASE("util/Parameters/toConfiguration") {
                      testToConfiguration();
                    });

    // Test fails because of a bug in the eckit YAML parser
    // ts.emplace_back(CASE("util/Parameters/mapParametersYamlStyleQuotedKeys") {
    //                   testMapParametersYamlStyleQuotedKeys();
    //                 });
    // Test fails because of a bug in the eckit YAML parser
    // ts.emplace_back(CASE("util/Parameters/mapParametersYamlStyleUnquotedKeys") {
    //                   testMapParametersYamlStyleUnquotedKeys();
    //                 });
    ts.emplace_back(CASE("util/Parameters/mapParametersJsonStyleQuotedKeys") {
                      testMapParametersJsonStyleQuotedKeys();
                    });
    // Test fails because of a bug in the eckit YAML parser
    // ts.emplace_back(CASE("util/Parameters/mapParametersJsonStyleUnquotedKeys") {
    //                   testMapParametersJsonStyleUnquotedKeys();
    //                 });

    ts.emplace_back(CASE("util/Parameters/mapParametersSerialization") {
                      testMapParametersSerialization();
                    });

    ts.emplace_back(CASE("util/Parameters/testSetIntParameters") {
                      testSetIntParameters();
                    });

    ts.emplace_back(CASE("util/Parameters/testVariablesDeserializationWithoutChannels") {
                      testVariablesDeserializationWithoutChannels();
                    });
    ts.emplace_back(CASE("util/Parameters/testVariablesDeserializationWithChannels") {
                      testVariablesDeserializationWithChannels();
                    });
    ts.emplace_back(CASE("util/Parameters/testVariablesSerializationWithoutChannels") {
                      testVariablesSerializationWithoutChannels();
                    });
    ts.emplace_back(CASE("util/Parameters/testVariablesSerializationWithChannels") {
                      testVariablesSerializationWithChannels();
                    });
    ts.emplace_back(CASE("util/Parameters/testCompositeVariablesSerialization") {
                      testCompositeVariablesSerialization();
                    });

    ts.emplace_back(CASE("util/Parameters/testPolymorphicParametersDeserialization") {
                      testPolymorphicParametersDeserialization();
                    });
    ts.emplace_back(CASE("util/Parameters/testPolymorphicParametersIncompleteDeserialization") {
                      testPolymorphicParametersIncompleteDeserialization();
                    });
    ts.emplace_back(CASE("util/Parameters/testOptionalPolymorphicParameters"
                         "IncompleteDeserialization") {
                      testOptionalPolymorphicParametersIncompleteDeserialization();
                    });
    ts.emplace_back(CASE("util/Parameters/testRequiredPolymorphicParameters"
                         "IncompleteDeserialization") {
                      testRequiredPolymorphicParametersIncompleteDeserialization();
                    });
    ts.emplace_back(CASE("util/Parameters/testMisspelledPropertyOfPolymorphicParameters") {
                      testMisspelledPropertyOfPolymorphicParameters();
                    });
    ts.emplace_back(CASE("util/Parameters/testMisspelledPropertyOfOptionalPolymorphicParameters") {
                      testMisspelledPropertyOfOptionalPolymorphicParameters();
                    });
    ts.emplace_back(CASE("util/Parameters/testMisspelledPropertyOfRequiredPolymorphicParameters") {
                      testMisspelledPropertyOfRequiredPolymorphicParameters();
                    });
    ts.emplace_back(CASE("util/Parameters/testInvalidPolymorphicParametersId") {
                      testInvalidPolymorphicParametersId();
                    });
    ts.emplace_back(CASE("util/Parameters/testInvalidOptionalPolymorphicParametersId") {
                      testInvalidOptionalPolymorphicParametersId();
                    });
    ts.emplace_back(CASE("util/Parameters/testInvalidRequiredPolymorphicParametersId") {
                      testInvalidRequiredPolymorphicParametersId();
                    });
    ts.emplace_back(CASE("util/Parameters/testPolymorphicParametersCopyConstructor") {
                      testPolymorphicParametersCopyConstructor();
                    });
    ts.emplace_back(CASE("util/Parameters/testPolymorphicParametersMoveConstructor") {
                      testPolymorphicParametersMoveConstructor();
                    });
    ts.emplace_back(CASE("util/Parameters/testPolymorphicParametersCopyAssignmentOperator") {
                      testPolymorphicParametersCopyAssignmentOperator();
                    });
    ts.emplace_back(CASE("util/Parameters/testPolymorphicParametersMoveAssignmentOperator") {
                      testPolymorphicParametersMoveAssignmentOperator();
                    });
    ts.emplace_back(CASE("util/Parameters/testPolymorphicParametersSerialization") {
                      testPolymorphicParametersSerialization();
                    });

    ts.emplace_back(CASE("util/Parameters/testParametersWithMinConstraint") {
                      testParametersWithMinConstraint();
                    });
    ts.emplace_back(CASE("util/Parameters/testRequiredParametersWithMinConstraint") {
                      testRequiredParametersWithMinConstraint();
                    });
    ts.emplace_back(CASE("util/Parameters/testOptionalParametersWithMinConstraint") {
                      testOptionalParametersWithMinConstraint();
                    });

    ts.emplace_back(CASE("util/Parameters/testParametersWithExclusiveMinConstraint") {
                      testParametersWithExclusiveMinConstraint();
                    });
    ts.emplace_back(CASE("util/Parameters/testRequiredParametersWithExclusiveMinConstraint") {
                      testRequiredParametersWithExclusiveMinConstraint();
                    });
    ts.emplace_back(CASE("util/Parameters/testOptionalParametersWithExclusiveMinConstraint") {
                      testOptionalParametersWithExclusiveMinConstraint();
                    });

    ts.emplace_back(CASE("util/Parameters/testParametersWithMaxConstraint") {
                      testParametersWithMaxConstraint();
                    });
    ts.emplace_back(CASE("util/Parameters/testRequiredParametersWithMaxConstraint") {
                      testRequiredParametersWithMaxConstraint();
                    });
    ts.emplace_back(CASE("util/Parameters/testOptionalParametersWithMaxConstraint") {
                      testOptionalParametersWithMaxConstraint();
                    });

    ts.emplace_back(CASE("util/Parameters/testParametersWithExclusiveMaxConstraint") {
                      testParametersWithExclusiveMaxConstraint();
                    });
    ts.emplace_back(CASE("util/Parameters/testRequiredParametersWithExclusiveMaxConstraint") {
                      testRequiredParametersWithExclusiveMaxConstraint();
                    });
    ts.emplace_back(CASE("util/Parameters/testOptionalParametersWithExclusiveMaxConstraint") {
                      testOptionalParametersWithExclusiveMaxConstraint();
                    });

    ts.emplace_back(CASE("util/Parameters/testConfigurationParameter") {
                      testConfigurationParameter();
                    });

    ts.emplace_back(CASE("util/Parameters/testSchemaConflictDetection") {
                      testSchemaConflictDetection();
                    });

    ts.emplace_back(CASE("util/Parameters/testIgnoreOtherParameters") {
                      testIgnoreOtherParameters();
                    });

    ts.emplace_back(CASE("util/Parameters/testValidateAndDeserialize") {
                      testValidateAndDeserialize();
                    });

    ts.emplace_back(CASE("util/Parameters/testHasParameters_") {
                      testHasParameters_();
                    });
  }

  void clear() const override {}
};

}  // namespace test

#endif  // TEST_UTIL_PARAMETERS_H_
