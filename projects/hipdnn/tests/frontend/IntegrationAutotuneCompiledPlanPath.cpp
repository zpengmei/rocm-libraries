// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration tests for the compiled-plan autotune path:
// create_execution_plans() -> build_plans(ALL) -> autotune() -> execute().
// Also tests the manual benchmark loop using plan-indexed access APIs.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include "AutotuneIntegrationFixture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

// NOTE (Decision C / OOS): CompiledPlanAutotuneExhaustiveBlocked (rocmlibs1 61-84) is
// DROPPED — it sets config.mode = TuneMode::EXHAUSTIVE (enum value removed) and asserts
// the reject guard (also removed). The two *EngineIdFilter tests (rocmlibs1 291-365) are
// OOS (member 8 / plan-spec path). The follow-on PR re-adds them.

namespace
{

using IntegrationAutotuneCompiledPlanPath = hipdnn_tests::AutotuneIntegrationFixture;

// Test: compiled-plan path autotune end-to-end workflow
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneEndToEnd)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    ASSERT_GE(maxWs, 0);

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    assertAnySucceeded(results, "No engine succeeded during compiled-plan autotune");

    buildWorkspaceAndExecute(bundle);
}

// Test: manual benchmark loop using plan-indexed access APIs
TEST_F(IntegrationAutotuneCompiledPlanPath, ManualLoopEndToEnd)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GT(planCount, 0);

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    int64_t bestIndex = -1;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto execResult
            = bundle.graph->execute_plan_at_index(_handle, bundle.variantPack, workspace.get(), i);
        if(execResult.is_good())
        {
            bestIndex = i;
            break;
        }
    }

    ASSERT_GE(bestIndex, 0) << "No plan succeeded during manual loop";

    result = bundle.graph->build_plan_at_index(bestIndex);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    buildWorkspaceAndExecute(bundle);
}

// Test: multi-engine plugin provides multiple plans and autotune benchmarks them
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneMultipleEngines)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // With multi-engine plugin, should have multiple plans
    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GE(planCount, 2) << "Multi-engine plugin should provide >= 2 plans";

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Multiple engines should produce multiple results
    ASSERT_GE(results.size(), 2u) << "Autotune should benchmark >= 2 engines";

    assertAnySucceeded(results, "At least one engine should succeed");

    buildWorkspaceAndExecute(bundle);
}

// Test: build_plans(ALL) compiles multiple engines and they are executable
TEST_F(IntegrationAutotuneCompiledPlanPath, BuildPlansAllCompilesMultipleEngines)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GE(planCount, 2);

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Verify multiple plans are actually executable
    int successCount = 0;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto execResult
            = bundle.graph->execute_plan_at_index(_handle, bundle.variantPack, workspace.get(), i);
        if(execResult.is_good())
        {
            ++successCount;
        }
    }
    ASSERT_GE(successCount, 2) << "build_plans(ALL) should compile multiple executable plans";
}

// Test: build_plans(ALL) compiles more plans than HEURISTICS_CHOICE
TEST_F(IntegrationAutotuneCompiledPlanPath, BuildPlansAllVsHeuristicsChoice)
{
    // Graph A: build_plans(HEURISTICS_CHOICE) — only compiles the active plan
    ConvGraphBundle bundleA;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundleA);
    auto result = bundleA.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = bundleA.graph->build_plans(BuildPlanPolicy::HEURISTICS_CHOICE);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCountA = bundleA.graph->get_execution_plan_count();
    ASSERT_GE(planCountA, 2) << "Multi-engine plugin should provide >= 2 plans";

    const int64_t maxWsA = bundleA.graph->get_autotune_workspace_size();
    const Workspace workspaceA(static_cast<size_t>(maxWsA));

    int successCountA = 0;
    for(int64_t i = 0; i < planCountA; ++i)
    {
        auto execResult = bundleA.graph->execute_plan_at_index(
            _handle, bundleA.variantPack, workspaceA.get(), i);
        if(execResult.is_good())
        {
            ++successCountA;
        }
    }

    // Graph B: build_plans(ALL) — compiles all plans
    ConvGraphBundle bundleB;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundleB);
    result = bundleB.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = bundleB.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCountB = bundleB.graph->get_execution_plan_count();
    const int64_t maxWsB = bundleB.graph->get_autotune_workspace_size();
    const Workspace workspaceB(static_cast<size_t>(maxWsB));

    int successCountB = 0;
    for(int64_t i = 0; i < planCountB; ++i)
    {
        auto execResult = bundleB.graph->execute_plan_at_index(
            _handle, bundleB.variantPack, workspaceB.get(), i);
        if(execResult.is_good())
        {
            ++successCountB;
        }
    }

    // ALL should compile more plans than HEURISTICS_CHOICE
    ASSERT_GT(successCountB, successCountA)
        << "build_plans(ALL) should compile more plans than HEURISTICS_CHOICE";
}

// Test: manual benchmark loop iterates over multiple engines
TEST_F(IntegrationAutotuneCompiledPlanPath, ManualLoopMultipleEngines)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GE(planCount, 2) << "Multi-engine plugin should provide >= 2 plans";

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Iterate all plans, track all successes
    std::vector<int64_t> successIndices;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto execResult
            = bundle.graph->execute_plan_at_index(_handle, bundle.variantPack, workspace.get(), i);
        if(execResult.is_good())
        {
            successIndices.push_back(i);
        }
    }

    ASSERT_GE(successIndices.size(), 2u) << "At least 2 plans should succeed during manual loop";

    // Select a non-zero index if possible
    const int64_t selectedIndex = successIndices.back();

    result = bundle.graph->build_plan_at_index(selectedIndex);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    buildWorkspaceAndExecute(bundle);
}

} // namespace
