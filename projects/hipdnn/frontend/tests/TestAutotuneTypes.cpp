// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/BenchmarkStatistics.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace hipdnn_frontend;

// NOTE (Decision D): the PlanSpec deduplication block (rocmlibs1 23-117), the
// CartesianProduct block (123-235), and the EngineVariant/KnobSweepAxis/EngineSweepSpec
// construction tests (466-506) are OUT-of-scope (plan-spec path); PlanSpec.hpp,
// CartesianProduct.hpp, and KnobConstants.hpp are not ported. AutotuneRankingFnIsCallable
// (427-444) is OOS (member 9). The follow-on PR re-adds these with add_engine_*().

// ============================================================================
// BenchmarkStatistics Tests
// ============================================================================

TEST(TestAutotuneTypes, MeanSingleValue)
{
    const std::vector<float> values = {5.0f};
    EXPECT_FLOAT_EQ(autotune::computeMean(values), 5.0f);
}

TEST(TestAutotuneTypes, MeanMultipleValues)
{
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    EXPECT_FLOAT_EQ(autotune::computeMean(values), 3.0f);
}

TEST(TestAutotuneTypes, MeanDoubleValues)
{
    const std::vector<double> values = {2.0, 4.0, 6.0};
    EXPECT_DOUBLE_EQ(autotune::computeMean(values), 4.0);
}

TEST(TestAutotuneTypes, MeanThrowsOnEmpty)
{
    const std::vector<float> values;
    EXPECT_THROW(autotune::computeMean(values), std::invalid_argument);
}

TEST(TestAutotuneTypes, StddevUniformValues)
{
    // All identical values should have zero standard deviation
    const std::vector<float> values = {3.0f, 3.0f, 3.0f, 3.0f};
    EXPECT_FLOAT_EQ(autotune::computeStddev(values), 0.0f);
}

TEST(TestAutotuneTypes, StddevKnownValues)
{
    // Population stddev of {2, 4, 4, 4, 5, 5, 7, 9}
    // Mean = 40/8 = 5.0
    // Variance = ((2-5)^2 + (4-5)^2 + (4-5)^2 + (4-5)^2 + (5-5)^2 + (5-5)^2 + (7-5)^2 +
    // (9-5)^2) / 8
    //          = (9+1+1+1+0+0+4+16)/8 = 32/8 = 4.0
    // Stddev = sqrt(4) = 2.0
    const std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    EXPECT_DOUBLE_EQ(autotune::computeStddev(values), 2.0);
}

TEST(TestAutotuneTypes, StddevThrowsOnEmpty)
{
    const std::vector<float> values;
    EXPECT_THROW(autotune::computeStddev(values), std::invalid_argument);
}

TEST(TestAutotuneTypes, CoVKnownValues)
{
    // Mean = 5.0, Stddev = 2.0, CoV = 2.0/5.0 = 0.4
    const std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    EXPECT_DOUBLE_EQ(autotune::computeCoefficientOfVariation(values), 0.4);
}

TEST(TestAutotuneTypes, CoVUniformValuesIsZero)
{
    const std::vector<float> values = {7.0f, 7.0f, 7.0f};
    EXPECT_FLOAT_EQ(autotune::computeCoefficientOfVariation(values), 0.0f);
}

TEST(TestAutotuneTypes, CoVAllZerosIsZero)
{
    // When mean is 0, CoV returns 0 to avoid division by zero
    const std::vector<float> values = {0.0f, 0.0f, 0.0f};
    EXPECT_FLOAT_EQ(autotune::computeCoefficientOfVariation(values), 0.0f);
}

TEST(TestAutotuneTypes, CoVThrowsOnEmpty)
{
    const std::vector<double> values;
    EXPECT_THROW(autotune::computeCoefficientOfVariation(values), std::invalid_argument);
}

TEST(TestAutotuneTypes, StddevSingleValue)
{
    // Single value: stddev = 0
    const std::vector<float> values = {42.0f};
    EXPECT_FLOAT_EQ(autotune::computeStddev(values), 0.0f);
}

// ============================================================================
// AutotuneConfig Default Values Tests
// ============================================================================

TEST(TestAutotuneTypes, AutotuneConfigCustomValues)
{
    // MIXED-edit (ii)#2: members 8 (engineIdFilter) and 10 (continueOnPrimingFailure)
    // are out of the subset (members 1-7 only); their set+assert pairs are dropped.
    // Decision C: TuneMode::EXHAUSTIVE is no longer a valid enum value → use AUTO.
    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.warmupIterations = 5;
    config.timedIterations = 20;
    config.maxIterations = 200;
    config.windowSize = 10;
    config.stabilityThreshold = 0.02f;

    EXPECT_EQ(config.mode, TuneMode::AUTO);
    EXPECT_EQ(config.strategy, AutotuneStrategy::RUN_UNTIL_STABLE);
    EXPECT_EQ(config.warmupIterations, 5);
    EXPECT_EQ(config.timedIterations, 20);
    EXPECT_EQ(config.maxIterations, 200);
    EXPECT_EQ(config.windowSize, 10);
    EXPECT_FLOAT_EQ(config.stabilityThreshold, 0.02f);
}

// ============================================================================
// AutotuneResult Tests
// ============================================================================

TEST(TestAutotuneTypes, AutotuneResultDefaults)
{
    const AutotuneResult result;
    EXPECT_EQ(result.engineId, -1);
    EXPECT_TRUE(result.engineName.empty());
    EXPECT_TRUE(result.knobSettings.empty());
    EXPECT_EQ(result.rank, -1);
    EXPECT_FLOAT_EQ(result.minTimeMs, 0.0f);
    EXPECT_FLOAT_EQ(result.avgTimeMs, 0.0f);
    EXPECT_FLOAT_EQ(result.stddevMs, 0.0f);
    EXPECT_EQ(result.iterationsRun, 0);
    EXPECT_FALSE(result.succeeded);
    EXPECT_EQ(result.modeUsed, TuneMode::AUTO);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.workspaceSize, 0);
    EXPECT_FALSE(result.ranExhaustive);
    EXPECT_TRUE(result.errorMessage.empty());
    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::RUN_UNTIL_STABLE);
}

TEST(TestAutotuneTypes, AutotuneResultPopulated)
{
    AutotuneResult result;
    result.engineId = 42;
    result.engineName = "MIOpen_ConvFwd";
    result.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    result.rank = 0;
    result.minTimeMs = 1.5f;
    result.avgTimeMs = 1.8f;
    result.stddevMs = 0.3f;
    result.iterationsRun = 10;
    result.succeeded = true;
    // Decision C: EXHAUSTIVE enum value dropped → use AUTO. ranExhaustive is an
    // independent bool and is kept set true to exercise the field round-trip.
    result.modeUsed = TuneMode::AUTO;
    result.converged = true;
    result.workspaceSize = 4096;
    result.ranExhaustive = true;
    result.strategyUsed = AutotuneStrategy::SINGLE_SHOT;

    EXPECT_EQ(result.engineId, 42);
    EXPECT_EQ(result.engineName, "MIOpen_ConvFwd");
    EXPECT_EQ(result.knobSettings.size(), 1u);
    EXPECT_EQ(result.rank, 0);
    EXPECT_FLOAT_EQ(result.minTimeMs, 1.5f);
    EXPECT_FLOAT_EQ(result.avgTimeMs, 1.8f);
    EXPECT_FLOAT_EQ(result.stddevMs, 0.3f);
    EXPECT_EQ(result.iterationsRun, 10);
    EXPECT_TRUE(result.succeeded);
    EXPECT_EQ(result.modeUsed, TuneMode::AUTO);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.workspaceSize, 4096);
    EXPECT_TRUE(result.ranExhaustive);
    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::SINGLE_SHOT);
}

// ============================================================================
// AutotuneStorageConfig Tests
// ============================================================================

TEST(TestAutotuneTypes, AutotuneStorageConfigCustomValues)
{
    AutotuneStorageConfig config;
    config.filePath = "/tmp/autotune_results.json";
    config.deleteAllExistingFileContent = true;

    EXPECT_EQ(config.filePath.string(), "/tmp/autotune_results.json");
    EXPECT_TRUE(config.deleteAllExistingFileContent);
}

// ============================================================================
// TuneMode and AutotuneStrategy Enum Tests
//
// NOTE (Decision C): TuneModeValues (rocmlibs1 450-453) is DROPPED — its sole
// assertion EXPECT_NE(TuneMode::AUTO, TuneMode::EXHAUSTIVE) no longer compiles once
// the EXHAUSTIVE enumerator is removed (TuneMode is reduced to { AUTO }).
// ============================================================================

TEST(TestAutotuneTypes, AutotuneStrategyValues)
{
    EXPECT_NE(AutotuneStrategy::SINGLE_SHOT, AutotuneStrategy::FIXED_AVERAGE);
    EXPECT_NE(AutotuneStrategy::FIXED_AVERAGE, AutotuneStrategy::RUN_UNTIL_STABLE);
    EXPECT_NE(AutotuneStrategy::SINGLE_SHOT, AutotuneStrategy::RUN_UNTIL_STABLE);
}
