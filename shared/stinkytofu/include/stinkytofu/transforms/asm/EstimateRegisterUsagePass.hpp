/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <cstdint>
#include <memory>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"

namespace stinkytofu {
class Pass;
class Function;
class PassContext;

/// Peak register usage estimate for a single kernel/function.
///
/// `peak*` values are the maximum number of simultaneously-live physical
/// register *indices* anywhere in the function (computed via a standard
/// backward dataflow on the CFG built by CFGBuilderPass).
///
/// `maxIdx*` values are `max(reg.idx + reg.num)` over every operand that
/// appears as a def/use anywhere in the function — i.e. the smallest
/// `amdhsa_next_free_*` value that would still be correctness-safe if the
/// kernel were emitted as-is.
///
/// `declared*` values are read from Function metadata (key
/// "Signature.nextFree*"), set by callers that have access to the
/// SignatureBase. Zero when the caller did not plumb them.
struct RegisterUsageEstimate {
    uint32_t peakVgpr = 0;
    uint32_t peakSgpr = 0;
    uint32_t maxVgprIdx = 0;
    uint32_t maxSgprIdx = 0;
    uint32_t declaredVgpr = 0;
    uint32_t declaredSgpr = 0;
};

/// Metadata keys written to the Function by the pass.
inline constexpr const char* kPeakVgprMetadataKey = "EstimateRegisterUsagePass.peakVgpr";
inline constexpr const char* kPeakSgprMetadataKey = "EstimateRegisterUsagePass.peakSgpr";
inline constexpr const char* kMaxVgprIdxMetadataKey = "EstimateRegisterUsagePass.maxVgprIdx";
inline constexpr const char* kMaxSgprIdxMetadataKey = "EstimateRegisterUsagePass.maxSgprIdx";

/// Metadata keys read by the pass (set by the caller that owns the signature).
inline constexpr const char* kDeclaredVgprMetadataKey = "Signature.nextFreeVgpr";
inline constexpr const char* kDeclaredSgprMetadataKey = "Signature.nextFreeSgpr";

/// Optional GEMM kernel parameters used purely for human-readable reporting
/// (MacroTile / MatrixInstruction / LDS bytes lines). All values are uint64;
/// absence (std::nullopt) means "unknown" and the report omits the line.
inline constexpr const char* kGemmMatrixInstMMetadataKey  = "Gemm.matrixInstM";
inline constexpr const char* kGemmMatrixInstNMetadataKey  = "Gemm.matrixInstN";
inline constexpr const char* kGemmMatrixInstKMetadataKey  = "Gemm.matrixInstK";
inline constexpr const char* kGemmMatrixInstBMetadataKey  = "Gemm.matrixInstB";
inline constexpr const char* kGemmMIWaveTile0MetadataKey  = "Gemm.miWaveTile0";
inline constexpr const char* kGemmMIWaveTile1MetadataKey  = "Gemm.miWaveTile1";
inline constexpr const char* kGemmMIWaveGroup0MetadataKey = "Gemm.miWaveGroup0";
inline constexpr const char* kGemmMIWaveGroup1MetadataKey = "Gemm.miWaveGroup1";
inline constexpr const char* kGemmDepthUMetadataKey       = "Gemm.depthU";
inline constexpr const char* kGemmLdsBytesMetadataKey     = "Gemm.ldsBytes";

/// Analysis result so other passes can read the estimate via AnalysisManager.
struct EstimateRegisterUsageAnalysis {
    STINKYTOFU_ANALYSIS_KEY("EstimateRegisterUsageAnalysis")
    using Result = RegisterUsageEstimate;
    static STINKYTOFU_EXPORT Result run(Function& F, AnalysisManager& AM);
};

/// Pass factory. The pass:
///   1. Reads kDeclaredVgprMetadataKey / kDeclaredSgprMetadataKey from the
///      Function if present.
///   2. Computes peak-live VGPR/SGPR via backward dataflow on the CFG.
///   3. Computes max-touched-index footprint.
///   4. Writes the four kPeak*/kMaxIdx* metadata keys back to the Function.
///   5. Prints a human-readable summary to std::cerr.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createEstimateRegisterUsagePass();

/// Convenience: compute peak register usage for a function without going
/// through the pass manager.
STINKYTOFU_EXPORT RegisterUsageEstimate
estimateRegisterUsage(Function& func, PassContext& passCtx);

}  // namespace stinkytofu
