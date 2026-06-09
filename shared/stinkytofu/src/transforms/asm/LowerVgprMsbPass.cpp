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
#include "stinkytofu/transforms/asm/LowerVgprMsbPass.hpp"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace {

void encodeVgprOperand(StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::Register) return;
    if (reg.reg.type != RegType::V) return;
    int msb = static_cast<int>(reg.reg.idx) / 256;
    if (msb == 0) return;
    int wantOffset = -msb * 256;
    if (reg.reg.offset == wantOffset) return;
    reg.reg.offset = static_cast<int16_t>(wantOffset);
}

class LowerVgprMsbPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "LowerVgprMsbPass";
    }

    Pass::ID getPassID() const override {
        return &LowerVgprMsbPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& /*passCtx*/,
                          AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            for (auto& irNode : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&irNode);
                if (!inst) continue;
                for (auto& src : const_cast<std::vector<StinkyRegister>&>(inst->getSrcRegs()))
                    encodeVgprOperand(src);
                for (auto& dst : const_cast<std::vector<StinkyRegister>&>(inst->getDestRegs()))
                    encodeVgprOperand(dst);
            }
        }
        return preserveCFGAnalyses();
    }
};

char LowerVgprMsbPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createLowerVgprMsbPass() {
    return std::make_unique<LowerVgprMsbPassImpl>();
}

}  // namespace stinkytofu
