// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file AutotuneTypes.hpp
 * @brief Core type definitions for the hipDNN autotuning system
 *
 * Defines the tuning mode and strategy enumerations, the AutotuneConfig
 * struct for controlling autotuning behavior, the AutotuneResult struct
 * for per-engine benchmarking results, and the AutotuneStorageConfig
 * struct for config file output parameters.
 */

#pragma once

#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace hipdnn_frontend
{

/**
 * @enum TuneMode
 * @brief Controls whether autotune() performs engine-internal cache priming
 *
 * AUTO mode benchmarks engines as-is.
 */
enum class TuneMode
{
    // Strict-subset cut site 4 (Decision C): the EXHAUSTIVE enumerator is dropped
    // from the cuDNN-equivalence subset (cuDNN has no equivalent knob). The
    // TuneMode concept and its AUTO value are retained. The follow-on PR re-adds
    // the EXHAUSTIVE enumerator with the plan-spec/priming path.
    AUTO ///< Simple wall-time comparison (no engine-internal cache priming)
};

/**
 * @enum AutotuneStrategy
 * @brief Benchmarking iteration strategy for timed runs
 *
 * Controls how many timed iterations are executed per engine and how
 * timing stability is assessed.
 */
enum class AutotuneStrategy
{
    SINGLE_SHOT, ///< 1 timed run, take the result
    FIXED_AVERAGE, ///< Average of N runs
    RUN_UNTIL_STABLE ///< Run until timing variance stabilizes, up to a cap (default)
};

/// Get the string representation of a TuneMode
inline const char* tuneModeToString(TuneMode mode)
{
    switch(mode)
    {
    case TuneMode::AUTO:
        return "AUTO";
    // Strict-subset cut site 5 (Decision C): the case TuneMode::EXHAUSTIVE arm is
    // dropped with the enumerator. The default arm keeps the switch sane.
    default:
        return "UNKNOWN";
    }
}

/// Get the string representation of an AutotuneStrategy
inline const char* strategyToString(AutotuneStrategy strategy)
{
    switch(strategy)
    {
    case AutotuneStrategy::SINGLE_SHOT:
        return "SINGLE_SHOT";
    case AutotuneStrategy::FIXED_AVERAGE:
        return "FIXED_AVERAGE";
    case AutotuneStrategy::RUN_UNTIL_STABLE:
        return "RUN_UNTIL_STABLE";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Per-engine benchmarking result from autotune()
 *
 * Contains timing data, ranking information, and status for each
 * engine configuration that was benchmarked (or attempted).
 */
struct AutotuneResult
{
    // ── Identity ───────────────────────────────────────────────────────
    int64_t engineId = -1; ///< Engine that was benchmarked
    std::string engineName; ///< Human-readable engine name

    /// Informational, records knobs explicitly set on the engine.
    std::vector<KnobSetting> knobSettings;

    // ── Timing ─────────────────────────────────────────────────────────
    float minTimeMs = 0.0f; ///< Minimum time across iterations (used for default ranking)
    float avgTimeMs = 0.0f; ///< Average time across iterations
    float stddevMs = 0.0f; ///< Standard deviation of timing measurements (0.0 for SINGLE_SHOT)
    int iterationsRun = 0; ///< Actual number of timed iterations executed

    /// true for SINGLE_SHOT and FIXED_AVERAGE when all iterations completed
    /// successfully. false on benchmark failure (any strategy) or for
    /// RUN_UNTIL_STABLE when maxIterations was reached without convergence.
    /// Only meaningful for RUN_UNTIL_STABLE; for SINGLE_SHOT and
    /// FIXED_AVERAGE, the value is deterministic (true on success, false
    /// on failure).
    bool converged = false;

    // ── Status ─────────────────────────────────────────────────────────
    int rank = -1; ///< 0-based ranking (0 = fastest); -1 for failed engines
    bool succeeded = false; ///< Whether this engine succeeded benchmarking
    std::string errorMessage; ///< Empty if no error; describes failure otherwise

    int64_t workspaceSize = 0; ///< Workspace bytes used by this engine
    int64_t estimatedWorkspaceSize = 0; ///< Pre-compile workspace estimate from engine config
    int compiledPlanIndex = -1; ///< Index into compiled plans vector; used for winner selection

    TuneMode modeUsed = TuneMode::AUTO; ///< Which mode was used for this engine

    /// true if this engine was primed via a temporary benchmarking plan before
    /// timing. false if the engine does not support exhaustive priming or AUTO
    /// mode was used.
    bool ranExhaustive = false;

    AutotuneStrategy strategyUsed
        = AutotuneStrategy::RUN_UNTIL_STABLE; ///< Which strategy was used for this engine
};

/**
 * @brief Configuration parameters for autotuning
 *
 * Controls the tuning mode, benchmarking strategy, iteration counts, and
 * convergence parameters.
 *
 * @code{.cpp}
 * AutotuneConfig config;
 * config.mode = TuneMode::AUTO;
 * config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
 * config.timedIterations = 20;
 * graph.autotune(handle, variantPack, workspace, config);
 * @endcode
 */
struct AutotuneConfig
{
    TuneMode mode = TuneMode::AUTO; ///< Tuning mode (AUTO)
    AutotuneStrategy strategy
        = AutotuneStrategy::RUN_UNTIL_STABLE; ///< Benchmarking iteration strategy (cuDNN parity)

    int warmupIterations = 1; ///< Number of warmup iterations before timed runs (cuDNN parity)
    int timedIterations = 10; ///< Number of timed iterations for FIXED_AVERAGE

    /// Maximum iterations for RUN_UNTIL_STABLE (must be >= windowSize)
    int maxIterations = 100;

    /// Window size for convergence check in RUN_UNTIL_STABLE (must be >= 2)
    int windowSize = 3;

    /// Coefficient of variation threshold for RUN_UNTIL_STABLE convergence (e.g. 0.05 = 5%)
    float stabilityThreshold = 0.05f;

    // Decision B: members 8-10 (engineIdFilter, rankingFn, continueOnPrimingFailure)
    // and the AutotuneRankingFn type are dropped from the cuDNN-equivalence subset.
    // This is the type-definition side of the Decision-B cuts; the Graph.hpp
    // consumers of members 8/9 are removed at the eight intentional non-verbatim
    // sites in Stage 5. AutotuneConfig stays an aggregate so the follow-on PR
    // appends those members source-compatibly.
};

/**
 * @brief Config file output parameters for autotune results
 *
 * When filePath is non-empty, autotune() writes the ranked results
 * to a JSON file in heuristic config format. The file can be
 * loaded on subsequent runs via HIPDNN_HEUR_CONFIG_PATH.
 */
struct AutotuneStorageConfig
{
    /// Output file path (empty = no file output)
    std::filesystem::path filePath;

    /// When true, delete all existing file content before writing new results.
    /// When false, replace only matching (operation, tensors) entries.
    bool deleteAllExistingFileContent = false;
};

} // namespace hipdnn_frontend
