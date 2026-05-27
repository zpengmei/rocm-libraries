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
#include "stinkytofu/transforms/asm/EstimateRegisterUsagePass.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

#define DEBUG_TYPE "EstimateRegisterUsagePass"

namespace {
using namespace stinkytofu;

/// VGPR pool covers V, ACC, AGPR (all share the same architectural budget /
/// amdhsa_next_free_vgpr field).
static bool isVgprPool(RegType t) {
    return t == RegType::V || t == RegType::ACC || t == RegType::AGPR;
}

/// SGPR pool covers only RegType::S. SCC/VCC/EXEC/M0 are not allocatable
/// and are excluded by isAllocatableReg().
static bool isSgprPool(RegType t) {
    return t == RegType::S;
}

/// Iterate every physical register index used (def or use) by an operand.
/// Skips literals, pseudo-registers, virtual registers (kVirtualBit), and
/// non-allocatable special registers (SCC/VCC/EXEC/M0).
template <class Fn>
static void forEachAllocatableIdx(const StinkyRegister& r, Fn&& fn) {
    if (r.dataType != StinkyRegister::Type::Register) return;
    if (r.isVirtualReg()) return;
    if (isPseudoReg(r)) return;
    if (!isAllocatableReg(r.reg.type)) return;
    const bool isV = isVgprPool(r.reg.type);
    const bool isS = isSgprPool(r.reg.type);
    if (!isV && !isS) return;
    for (uint16_t off = 0; off < r.reg.num; ++off) {
        fn(r.reg.idx + off, isV);
    }
}

struct PerBBLiveSets {
    // Standard dataflow sets, separated by pool. Indices in these sets are
    // *physical* register indices (already filtered through forEachAllocatableIdx).
    std::unordered_set<uint32_t> vgprUse;
    std::unordered_set<uint32_t> vgprDef;
    std::unordered_set<uint32_t> sgprUse;
    std::unordered_set<uint32_t> sgprDef;
    std::unordered_set<uint32_t> vgprLiveIn;
    std::unordered_set<uint32_t> vgprLiveOut;
    std::unordered_set<uint32_t> sgprLiveIn;
    std::unordered_set<uint32_t> sgprLiveOut;
};

class EstimateRegisterUsagePassImpl : public Pass {
   public:
    static constexpr const char* PassName = "EstimateRegisterUsagePass";
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return PassName;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        (void)passCtx;
        (void)AM;

        RegisterUsageEstimate est = compute(func);
        if (auto v = func.getMetaData(kDeclaredVgprMetadataKey))
            est.declaredVgpr = static_cast<uint32_t>(*v);
        if (auto v = func.getMetaData(kDeclaredSgprMetadataKey))
            est.declaredSgpr = static_cast<uint32_t>(*v);

        func.setMetaData(kPeakVgprMetadataKey, est.peakVgpr);
        func.setMetaData(kPeakSgprMetadataKey, est.peakSgpr);
        func.setMetaData(kMaxVgprIdxMetadataKey, est.maxVgprIdx);
        func.setMetaData(kMaxSgprIdxMetadataKey, est.maxSgprIdx);

        report(func, est);
        return PreservedAnalyses::all();
    }

    /// Compute the estimate. Public so the free function estimateRegisterUsage
    /// can re-use it without going through the pass manager.
    static RegisterUsageEstimate compute(Function& func) {
        RegisterUsageEstimate est;

        if (func.empty()) return est;

        // Collect per-BB use/def sets and remember instruction lists.
        std::unordered_map<const BasicBlock*, PerBBLiveSets> info;
        std::unordered_map<const BasicBlock*, std::vector<StinkyInstruction*>> bbInsts;
        info.reserve(func.size());
        bbInsts.reserve(func.size());

        for (BasicBlock& bb : func) {
            PerBBLiveSets& sets = info[&bb];
            std::vector<StinkyInstruction*>& insts = bbInsts[&bb];

            for (IRBase& ir : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (!inst) continue;
                insts.push_back(inst);
                accumulateUseDef(inst, sets, est);
            }
        }

        // Iterative backward dataflow:
        //   liveOut[B] = ∪ liveIn[S] for S in successors(B)
        //   liveIn [B] = use[B] ∪ (liveOut[B] − def[B])
        bool changed = true;
        size_t iter = 0;
        while (changed) {
            changed = false;
            ++iter;
            for (BasicBlock& bb : func) {
                PerBBLiveSets& s = info[&bb];

                // Recompute liveOut from successors' liveIn
                for (BasicBlock* succ : bb.getSuccessors()) {
                    auto it = info.find(succ);
                    if (it == info.end()) continue;
                    for (uint32_t idx : it->second.vgprLiveIn) {
                        if (s.vgprLiveOut.insert(idx).second) changed = true;
                    }
                    for (uint32_t idx : it->second.sgprLiveIn) {
                        if (s.sgprLiveOut.insert(idx).second) changed = true;
                    }
                }

                // Recompute liveIn = use ∪ (liveOut − def)
                for (uint32_t idx : s.vgprLiveOut) {
                    if (s.vgprDef.count(idx)) continue;
                    if (s.vgprLiveIn.insert(idx).second) changed = true;
                }
                for (uint32_t idx : s.sgprLiveOut) {
                    if (s.sgprDef.count(idx)) continue;
                    if (s.sgprLiveIn.insert(idx).second) changed = true;
                }
                // Seed liveIn with use[] (only matters first iteration).
                for (uint32_t idx : s.vgprUse) {
                    if (s.vgprLiveIn.insert(idx).second) changed = true;
                }
                for (uint32_t idx : s.sgprUse) {
                    if (s.sgprLiveIn.insert(idx).second) changed = true;
                }
            }
            // Safety: large kernels with very tangled CFG converge in a few
            // iterations; an upper bound guards against builder bugs.
            if (iter > 2000) {
                std::cerr << "[EstimateRegisterUsage] WARNING: dataflow did not converge after "
                          << iter << " iterations; aborting fixed-point\n";
                break;
            }
        }

        // Per-BB backward sweep to track in-block peak.
        for (BasicBlock& bb : func) {
            const PerBBLiveSets& s = info[&bb];
            std::unordered_set<uint32_t> vLive = s.vgprLiveOut;
            std::unordered_set<uint32_t> sLive = s.sgprLiveOut;

            est.peakVgpr = std::max(est.peakVgpr, static_cast<uint32_t>(vLive.size()));
            est.peakSgpr = std::max(est.peakSgpr, static_cast<uint32_t>(sLive.size()));

            const auto& insts = bbInsts[&bb];
            for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
                StinkyInstruction* inst = *it;
                // Step "above" this instruction:
                //   kill defs, then gen uses.
                // Peak is measured *before* the kill (the value being defined
                // is live during the def-cycle of this instruction).
                est.peakVgpr = std::max(est.peakVgpr, static_cast<uint32_t>(vLive.size()));
                est.peakSgpr = std::max(est.peakSgpr, static_cast<uint32_t>(sLive.size()));

                for (const auto& r : inst->getDestRegs()) {
                    forEachAllocatableIdx(r, [&](uint32_t idx, bool isV) {
                        if (isV) vLive.erase(idx);
                        else sLive.erase(idx);
                    });
                }
                for (const auto& r : inst->getSrcRegs()) {
                    forEachAllocatableIdx(r, [&](uint32_t idx, bool isV) {
                        if (isV) vLive.insert(idx);
                        else sLive.insert(idx);
                    });
                }
                est.peakVgpr = std::max(est.peakVgpr, static_cast<uint32_t>(vLive.size()));
                est.peakSgpr = std::max(est.peakSgpr, static_cast<uint32_t>(sLive.size()));
            }
        }

        return est;
    }

   private:
    static void accumulateUseDef(StinkyInstruction* inst, PerBBLiveSets& sets,
                                 RegisterUsageEstimate& est) {
        // Gen/kill semantics: a register is in use[BB] only if it is read
        // before being written within this BB. It is in def[BB] only if it
        // is written somewhere in this BB.

        // Process srcs first (uses), then dests (defs), matching execution
        // order so the "killed-then-used" pattern (e.g. v_add v0, v0, v1)
        // is classified correctly.
        for (const auto& r : inst->getSrcRegs()) {
            forEachAllocatableIdx(r, [&](uint32_t idx, bool isV) {
                auto& use = isV ? sets.vgprUse : sets.sgprUse;
                auto& def = isV ? sets.vgprDef : sets.sgprDef;
                if (!def.count(idx)) use.insert(idx);

                // maxIdx footprint
                if (isV) est.maxVgprIdx = std::max(est.maxVgprIdx, idx + 1);
                else est.maxSgprIdx = std::max(est.maxSgprIdx, idx + 1);
            });
        }
        for (const auto& r : inst->getDestRegs()) {
            forEachAllocatableIdx(r, [&](uint32_t idx, bool isV) {
                auto& def = isV ? sets.vgprDef : sets.sgprDef;
                def.insert(idx);

                if (isV) est.maxVgprIdx = std::max(est.maxVgprIdx, idx + 1);
                else est.maxSgprIdx = std::max(est.maxSgprIdx, idx + 1);
            });
        }
    }

    /// Emit "MacroTile=MT<MT0>x<MT1>x<MTK>", MatrixInstruction, and LDS bytes lines
    /// when the corresponding metadata keys are present on the Function. Each
    /// piece is independently optional: missing keys silently drop their line.
    /// MacroTile dims are derived as MatrixInst[M|N] * MIWaveTile[0|1] * MIWaveGroup[0|1];
    /// the K-dim is DepthU.
    static void reportGemmParams(const Function& func) {
        auto get = [&](const char* k) -> std::optional<uint64_t> {
            return func.getMetaData(k);
        };

        auto miM   = get(kGemmMatrixInstMMetadataKey);
        auto miN   = get(kGemmMatrixInstNMetadataKey);
        auto miK   = get(kGemmMatrixInstKMetadataKey);
        auto miB   = get(kGemmMatrixInstBMetadataKey);
        auto wt0   = get(kGemmMIWaveTile0MetadataKey);
        auto wt1   = get(kGemmMIWaveTile1MetadataKey);
        auto wg0   = get(kGemmMIWaveGroup0MetadataKey);
        auto wg1   = get(kGemmMIWaveGroup1MetadataKey);
        auto du    = get(kGemmDepthUMetadataKey);
        auto ldsB  = get(kGemmLdsBytesMetadataKey);

        if (miM && miN && wt0 && wt1 && wg0 && wg1 && du) {
            const uint64_t mt0 = (*miM) * (*wt0) * (*wg0);
            const uint64_t mt1 = (*miN) * (*wt1) * (*wg1);
            const uint64_t mtk = *du;
            std::cerr << "                        MacroTile=MT" << mt0 << "x" << mt1
                      << "x" << mtk
                      << "  (MIWaveTile=[" << *wt0 << "," << *wt1 << "]"
                      << "  MIWaveGroup=[" << *wg0 << "," << *wg1 << "]"
                      << "  DepthU=" << *du << ")\n";
        }

        if (miM && miN && miK && miB) {
            std::cerr << "                        MatrixInstruction=[" << *miM << ", " << *miN
                      << ", " << *miK << ", " << *miB << "]\n";
        }

        if (ldsB) {
            std::cerr << "                        LdsNumBytes=" << *ldsB << "\n";
        }
    }

    static void report(const Function& func, const RegisterUsageEstimate& est) {
        auto pct = [](uint32_t saved, uint32_t total) -> double {
            if (total == 0) return 0.0;
            return 100.0 * static_cast<double>(saved) / static_cast<double>(total);
        };

        std::cerr << "\n[EstimateRegisterUsage] kernel=\"" << func.getName() << "\"\n";
        std::cerr << "                        basicBlocks=" << func.size() << "\n";

        reportGemmParams(func);
        std::cerr << "  VGPR:  peakLive=" << est.peakVgpr
                  << "  maxIdxTouched=" << est.maxVgprIdx
                  << "  declared=" << est.declaredVgpr;
        if (est.declaredVgpr > 0) {
            int32_t metadataGap = static_cast<int32_t>(est.declaredVgpr) -
                                  static_cast<int32_t>(est.maxVgprIdx);
            int32_t raGain = static_cast<int32_t>(est.maxVgprIdx) -
                             static_cast<int32_t>(est.peakVgpr);
            int32_t totalGap = static_cast<int32_t>(est.declaredVgpr) -
                               static_cast<int32_t>(est.peakVgpr);
            std::cerr << "  metadataFix=" << metadataGap << "  raGain=" << raGain
                      << "  potentialSaving=" << totalGap << " ("
                      << pct(totalGap > 0 ? static_cast<uint32_t>(totalGap) : 0u,
                             est.declaredVgpr)
                      << "%)";
        }
        std::cerr << "\n";

        std::cerr << "  SGPR:  peakLive=" << est.peakSgpr
                  << "  maxIdxTouched=" << est.maxSgprIdx
                  << "  declared=" << est.declaredSgpr;
        if (est.declaredSgpr > 0) {
            int32_t metadataGap = static_cast<int32_t>(est.declaredSgpr) -
                                  static_cast<int32_t>(est.maxSgprIdx);
            int32_t raGain = static_cast<int32_t>(est.maxSgprIdx) -
                             static_cast<int32_t>(est.peakSgpr);
            int32_t totalGap = static_cast<int32_t>(est.declaredSgpr) -
                               static_cast<int32_t>(est.peakSgpr);
            std::cerr << "  metadataFix=" << metadataGap << "  raGain=" << raGain
                      << "  potentialSaving=" << totalGap << " ("
                      << pct(totalGap > 0 ? static_cast<uint32_t>(totalGap) : 0u,
                             est.declaredSgpr)
                      << "%)";
        }
        std::cerr << "\n\n";

        std::cerr << "  Legend:\n";
        std::cerr << "    peakLive      = peak simultaneously-live physical regs (CFG dataflow)\n";
        std::cerr
            << "    maxIdxTouched = max(reg.idx + reg.num) over all defs/uses; min legal\n";
        std::cerr << "                    amdhsa_next_free_* if emitted as-is\n";
        std::cerr
            << "    declared      = amdhsa_next_free_* read from the kernel descriptor\n";
        std::cerr << "    metadataFix   = declared - maxIdxTouched  (free win, no codegen change)\n";
        std::cerr << "    raGain        = maxIdxTouched - peakLive  (saving a real RA could give)\n";
        std::cerr
            << "    potentialSaving = declared - peakLive          (overall headroom)\n";
    }
};
char EstimateRegisterUsagePassImpl::ID = 0;

}  // namespace

namespace stinkytofu {

std::unique_ptr<Pass> createEstimateRegisterUsagePass() {
    return std::make_unique<EstimateRegisterUsagePassImpl>();
}

EstimateRegisterUsageAnalysis::Result EstimateRegisterUsageAnalysis::run(Function& F,
                                                                         AnalysisManager& AM) {
    (void)AM;
    RegisterUsageEstimate est = EstimateRegisterUsagePassImpl::compute(F);
    if (auto v = F.getMetaData(kDeclaredVgprMetadataKey))
        est.declaredVgpr = static_cast<uint32_t>(*v);
    if (auto v = F.getMetaData(kDeclaredSgprMetadataKey))
        est.declaredSgpr = static_cast<uint32_t>(*v);
    return est;
}

RegisterUsageEstimate estimateRegisterUsage(Function& func, PassContext& passCtx) {
    (void)passCtx;
    return EstimateRegisterUsagePassImpl::compute(func);
}

}  // namespace stinkytofu
