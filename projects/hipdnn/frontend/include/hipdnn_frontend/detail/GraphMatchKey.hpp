// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/LayerNormNode.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_frontend/node/NodeType.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/RMSNormNode.hpp>
#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_frontend/node/SdpaBwdNode.hpp>
#include <hipdnn_frontend/node/SdpaFwdNode.hpp>
#endif // HIPDNN_ENABLE_SDPA

#include <memory>
#include <string>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Determine the core operation name written to the autotune config file for a
/// graph (passed as its root INode). Priority: convolution/GEMM/SDPA (highest)
/// > normalization > pointwise (lowest); the graph name is the fallback when no
/// recognized operation is found. Reconstructed through the public node
/// traversal API (INode::visit + getNodeType + graph_attributes) so no friend
/// or new accessor is required.
inline std::string getCoreOperationName(const graph::INode& root)
{
    // Priority levels: 0 = default/unknown, 1 = pointwise, 2 = normalization,
    // 3 = conv/matmul/SDPA
    int bestPriority = 0;
    std::string bestName;

    root.visit([&](const graph::INode& node) {
        if(bestPriority == 3)
        {
            // Short-circuit: can't do better than priority 3.
            return;
        }

        int priority = 0;
        std::string name;
        switch(node.getNodeType())
        {
        case graph::NodeType::CONVOLUTION_FPROP:
            priority = 3;
            name = "conv_fprop";
            break;
        case graph::NodeType::CONVOLUTION_DGRAD:
            priority = 3;
            name = "conv_dgrad";
            break;
        case graph::NodeType::CONVOLUTION_WGRAD:
            priority = 3;
            name = "conv_wgrad";
            break;
        case graph::NodeType::MATMUL:
            priority = 3;
            name = "matmul";
            break;
        case graph::NodeType::SDPA_FWD:
            priority = 3;
            name = "sdpa_fwd";
            break;
        case graph::NodeType::SDPA_BWD:
            priority = 3;
            name = "sdpa_bwd";
            break;
        case graph::NodeType::BATCHNORM:
            priority = 2;
            name = "batchnorm_training";
            break;
        case graph::NodeType::BATCHNORM_INFERENCE:
            priority = 2;
            name = "batchnorm_inference";
            break;
        case graph::NodeType::BATCHNORM_BACKWARD:
            priority = 2;
            name = "batchnorm_backward";
            break;
        case graph::NodeType::LAYER_NORM:
            priority = 2;
            name = "layernorm";
            break;
        case graph::NodeType::RMS_NORM:
            priority = 2;
            name = "rmsnorm";
            break;
        case graph::NodeType::POINTWISE:
        {
            // Pointwise arity (and thus the op string) is a function of the mode,
            // derived the SAME way the backend reader derives it: the frontend
            // classifiers hipdnn_frontend::isUnaryPointwiseMode /
            // isBinaryPointwiseMode (Types.hpp) mirror the backend's
            // hipdnn_flatbuffers_sdk::utilities::* (PointwiseValidation.hpp), so
            // both sides agree on arity by construction. Ternary is intentionally
            // NOT emitted here yet (no factory builder / OperationType / functor);
            // a ternary node falls through to priority 0 and the graph-name
            // fallback below.
            const auto& pw = static_cast<const graph::PointwiseNode&>(node);
            const auto mode = pw.attributes.get_mode();
            if(isUnaryPointwiseMode(mode))
            {
                priority = 1;
                name = "pointwise_unary";
            }
            else if(isBinaryPointwiseMode(mode))
            {
                priority = 1;
                name = "pointwise_binary";
            }
            break;
        }
        default:
            break;
        }

        if(priority > bestPriority)
        {
            bestPriority = priority;
            bestName = std::move(name);
        }
    });

    return bestPriority > 0 ? bestName : std::string(root.graph_attributes.get_name());
}

/// Append a tensor to @p out when it participates in the match key: non-null,
/// has an assigned UID, and is not virtual (intermediate). Mirrors the
/// non-virtual / has_uid filter the backend index applies to physical tensors.
inline void appendMatchKeyTensor(std::vector<std::shared_ptr<graph::TensorAttributes>>& out,
                                 const std::shared_ptr<graph::TensorAttributes>& tensor)
{
    if(tensor && tensor->has_uid() && !tensor->get_is_virtual())
    {
        out.push_back(tensor);
    }
}

/// Collect the ordered match-key tensors for a graph (passed as its root INode)
/// written to the autotune config file.
///
/// CANONICAL-TENSOR-ORDER: for every operation the match-key tensor order is
/// the op's flatbuffer `*_attributes.fbs` INPUT-field declaration order with the
/// output field(s) dropped; the match is POSITIONAL (index-by-index). This rule
/// MUST stay in lockstep with the backend reader that consumes these tensors:
/// `hipdnn_backend::heuristics::config::matchOverrideConfig` (the conv dispatch)
/// in `backend/src/heuristics/config/ConfigBuiltIn.cpp`. Any change to the
/// per-op set/order here MUST be mirrored there and vice versa.
///
/// Convolution (output excluded, exactly 2 inputs each):
///   - conv_fprop → (x, w)
///   - conv_dgrad → (dy, w)
///   - conv_wgrad → (x, dy)
/// Matmul (output c excluded, exactly 2 inputs):
///   - matmul → (a, b)
/// SDPA forward (output o and all optional inputs excluded, 3 required inputs):
///   - sdpa_fwd → (q, k, v)
/// SDPA backward (output gradients dq/dk/dv and all optional inputs excluded,
/// 6 required inputs):
///   - sdpa_bwd → (q, k, v, o, dO, stats)
/// RMS normalization (output y and the optional bias/inv_rms fields excluded,
/// 3 required inputs; epsilon is a real UID-bearing value-tensor, not a dropped
/// scalar, so it is part of the key):
///   - rmsnorm → (x, scale, epsilon)
/// Layer normalization (outputs y/mean/inv_variance excluded, 4 required inputs;
/// epsilon is a real UID-bearing value-tensor, not a dropped scalar, so it is
/// part of the key):
///   - layernorm → (x, scale, bias, epsilon)
/// Batchnorm — one distinct op string per variant (NOT collapsed to a single
/// "batchnorm" key). Each variant excludes its outputs and any nullable/optional
/// inputs:
///   - batchnorm_inference → (x, mean, inv_variance, scale, bias)
///   - batchnorm_training  → (x, scale, bias, epsilon); epsilon is a real
///     UID-bearing value-tensor (dims {1}), like layernorm/rmsnorm
///   - batchnorm_backward  → (dy, x, scale); the optional mean/inv_variance are
///     built absent and excluded from the key
/// Pointwise — one distinct op string per ARITY (NOT collapsed to a single
/// "pointwise" key); arity is a function of the mode, classified the same way on
/// both sides. Each arity excludes the output (out_0):
///   - pointwise_unary  → (in_0)         — unary modes (e.g. RELU_FWD)
///   - pointwise_binary → (in_0, in_1)   — binary modes (e.g. ADD)
///
/// Each operation supported for config round-trip has an explicit op-aware
/// branch above; the branch defines its canonical match-key tensor set/order. An
/// operation with NO op-aware branch (e.g. a reduction, or a ternary pointwise)
/// is intentionally unsupported for config round-trip: it produces an EMPTY match
/// key and therefore never matches a backend config rule.
inline std::vector<std::shared_ptr<graph::TensorAttributes>>
    getMatchKeyTensors(const graph::INode& root)
{
    std::vector<std::shared_ptr<graph::TensorAttributes>> result;

    // Op-aware selection: the first node matching an op-aware branch wins and
    // emits its tensors in the canonical order above; `handled` then stops the
    // visit. A graph with no recognized op leaves `handled` false and `result`
    // empty.
    bool handled = false;
    root.visit([&](const graph::INode& node) {
        if(handled)
        {
            return;
        }
        switch(node.getNodeType())
        {
        case graph::NodeType::CONVOLUTION_FPROP:
        {
            const auto& conv = static_cast<const graph::ConvolutionFpropNode&>(node);
            appendMatchKeyTensor(result, conv.attributes.get_x());
            appendMatchKeyTensor(result, conv.attributes.get_w());
            handled = true;
            break;
        }
        case graph::NodeType::CONVOLUTION_DGRAD:
        {
            const auto& conv = static_cast<const graph::ConvolutionDgradNode&>(node);
            appendMatchKeyTensor(result, conv.attributes.get_dy());
            appendMatchKeyTensor(result, conv.attributes.get_w());
            handled = true;
            break;
        }
        case graph::NodeType::CONVOLUTION_WGRAD:
        {
            const auto& conv = static_cast<const graph::ConvolutionWgradNode&>(node);
            appendMatchKeyTensor(result, conv.attributes.get_x());
            appendMatchKeyTensor(result, conv.attributes.get_dy());
            handled = true;
            break;
        }
        // CANONICAL-TENSOR-ORDER: matmul match key is the matmul_attributes.fbs
        // INPUT-field declaration order (a, b); the output field c is dropped; the
        // match is POSITIONAL. This MUST stay in lockstep with the backend reader
        // that consumes these tensors: the matmul dispatch in
        // hipdnn_backend::heuristics::config::matchOverrideConfig in
        // backend/src/heuristics/config/ConfigBuiltIn.cpp. Any change to this
        // set/order MUST be mirrored there and vice versa.
        case graph::NodeType::MATMUL:
        {
            const auto& mm = static_cast<const graph::MatmulNode&>(node);
            appendMatchKeyTensor(result, mm.attributes.get_a());
            appendMatchKeyTensor(result, mm.attributes.get_b());
            handled = true;
            break;
        }
        // CANONICAL-TENSOR-ORDER: rmsnorm match key is the
        // rmsnorm_attributes.fbs INPUT-field declaration order (x, scale,
        // epsilon); the output field y and the optional bias/inv_rms fields are
        // dropped; the match is POSITIONAL. epsilon is a real UID-bearing
        // value-tensor (dims {1}), NOT a dropped scalar, so it is the third
        // positional input. This MUST stay in lockstep with the backend reader
        // that consumes these tensors: the rmsnorm dispatch in
        // hipdnn_backend::heuristics::config::matchOverrideConfig in
        // backend/src/heuristics/config/ConfigBuiltIn.cpp. Any change to this
        // set/order MUST be mirrored there and vice versa.
        case graph::NodeType::RMS_NORM:
        {
            const auto& rms = static_cast<const graph::RMSNormNode&>(node);
            appendMatchKeyTensor(result, rms.attributes.get_x());
            appendMatchKeyTensor(result, rms.attributes.get_scale());
            appendMatchKeyTensor(result, rms.attributes.get_epsilon());
            handled = true;
            break;
        }
        // CANONICAL-TENSOR-ORDER: layernorm match key is the
        // layernorm_attributes.fbs INPUT-field declaration order (x, scale, bias,
        // epsilon); the output fields y/mean/inv_variance and the scalar attrs
        // (normalized_dim_count, forward_phase) are dropped; the match is
        // POSITIONAL. epsilon is a real UID-bearing value-tensor (dims {1}), NOT a
        // dropped scalar, so it is the fourth positional input. This MUST stay in
        // lockstep with the backend reader that consumes these tensors: the
        // layernorm dispatch in
        // hipdnn_backend::heuristics::config::matchOverrideConfig in
        // backend/src/heuristics/config/ConfigBuiltIn.cpp. Any change to this
        // set/order MUST be mirrored there and vice versa.
        case graph::NodeType::LAYER_NORM:
        {
            const auto& ln = static_cast<const graph::LayerNormNode&>(node);
            appendMatchKeyTensor(result, ln.attributes.get_x());
            appendMatchKeyTensor(result, ln.attributes.get_scale());
            appendMatchKeyTensor(result, ln.attributes.get_bias());
            appendMatchKeyTensor(result, ln.attributes.get_epsilon());
            handled = true;
            break;
        }
        // CANONICAL-TENSOR-ORDER: batchnorm_inference match key is the
        // batchnorm_inference_attributes.fbs INPUT-field declaration order
        // (x, mean, inv_variance, scale, bias); the output field y is dropped;
        // the match is POSITIONAL. All five are real UID-bearing physical
        // buffers (no optionals). This MUST stay in lockstep with the backend
        // reader that consumes these tensors: the batchnorm_inference dispatch in
        // hipdnn_backend::heuristics::config::matchOverrideConfig in
        // backend/src/heuristics/config/ConfigBuiltIn.cpp. Any change to this
        // set/order MUST be mirrored there and vice versa.
        case graph::NodeType::BATCHNORM_INFERENCE:
        {
            const auto& bn = static_cast<const graph::BatchnormInferenceNode&>(node);
            appendMatchKeyTensor(result, bn.attributes.get_x());
            appendMatchKeyTensor(result, bn.attributes.get_mean());
            appendMatchKeyTensor(result, bn.attributes.get_inv_variance());
            appendMatchKeyTensor(result, bn.attributes.get_scale());
            appendMatchKeyTensor(result, bn.attributes.get_bias());
            handled = true;
            break;
        }
        // CANONICAL-TENSOR-ORDER: batchnorm_training match key is the
        // batchnorm_attributes.fbs INPUT-field declaration order
        // (x, scale, bias, epsilon); the output fields y/mean/inv_variance and
        // the nullable running-stats fields are dropped; the match is POSITIONAL.
        // epsilon is a real UID-bearing value-tensor (dims {1}), NOT a dropped
        // scalar, so it is the fourth positional input. This MUST stay in
        // lockstep with the backend reader that consumes these tensors: the
        // batchnorm_training dispatch in
        // hipdnn_backend::heuristics::config::matchOverrideConfig in
        // backend/src/heuristics/config/ConfigBuiltIn.cpp. Any change to this
        // set/order MUST be mirrored there and vice versa.
        case graph::NodeType::BATCHNORM:
        {
            const auto& bn = static_cast<const graph::BatchnormNode&>(node);
            appendMatchKeyTensor(result, bn.attributes.get_x());
            appendMatchKeyTensor(result, bn.attributes.get_scale());
            appendMatchKeyTensor(result, bn.attributes.get_bias());
            appendMatchKeyTensor(result, bn.attributes.get_epsilon());
            handled = true;
            break;
        }
        // CANONICAL-TENSOR-ORDER: batchnorm_backward match key is the
        // batchnorm_backward_attributes.fbs REQUIRED INPUT-field declaration
        // order (dy, x, scale); the output gradient fields (dx/dscale/dbias),
        // the nullable mean/inv_variance inputs, and the peer-stats vector are
        // dropped; the match is POSITIONAL. This MUST stay in lockstep with the
        // backend reader that consumes these tensors: the batchnorm_backward
        // dispatch in hipdnn_backend::heuristics::config::matchOverrideConfig in
        // backend/src/heuristics/config/ConfigBuiltIn.cpp. Any change to this
        // set/order MUST be mirrored there and vice versa.
        case graph::NodeType::BATCHNORM_BACKWARD:
        {
            const auto& bn = static_cast<const graph::BatchnormBackwardNode&>(node);
            appendMatchKeyTensor(result, bn.attributes.get_dy());
            appendMatchKeyTensor(result, bn.attributes.get_x());
            appendMatchKeyTensor(result, bn.attributes.get_scale());
            handled = true;
            break;
        }
#ifdef HIPDNN_ENABLE_SDPA
        // CANONICAL-TENSOR-ORDER: sdpa_fwd match key is the sdpa_attributes.fbs
        // REQUIRED INPUT fields (q, k, v) in declaration order; the output field o
        // and every optional input/output field are dropped; the match is
        // POSITIONAL. This MUST stay in lockstep with the backend reader that
        // consumes these tensors: the sdpa_fwd dispatch in
        // hipdnn_backend::heuristics::config::matchOverrideConfig in
        // backend/src/heuristics/config/ConfigBuiltIn.cpp. Any change to this
        // set/order MUST be mirrored there and vice versa.
        case graph::NodeType::SDPA_FWD:
        {
            const auto& sdpa = static_cast<const graph::SdpaFwdNode&>(node);
            appendMatchKeyTensor(result, sdpa.attributes.get_q());
            appendMatchKeyTensor(result, sdpa.attributes.get_k());
            appendMatchKeyTensor(result, sdpa.attributes.get_v());
            handled = true;
            break;
        }
        // CANONICAL-TENSOR-ORDER: sdpa_bwd match key is the
        // sdpa_backward_attributes.fbs REQUIRED INPUT fields (q, k, v, o, dO,
        // stats) in declaration order; the output gradient fields (dq, dk, dv)
        // and every optional input/output field are dropped; the match is
        // POSITIONAL. This MUST stay in lockstep with the backend reader that
        // consumes these tensors: the sdpa_bwd dispatch in
        // hipdnn_backend::heuristics::config::matchOverrideConfig in
        // backend/src/heuristics/config/ConfigBuiltIn.cpp. Any change to this
        // set/order MUST be mirrored there and vice versa.
        case graph::NodeType::SDPA_BWD:
        {
            const auto& sdpa = static_cast<const graph::SdpaBwdNode&>(node);
            appendMatchKeyTensor(result, sdpa.attributes.get_q());
            appendMatchKeyTensor(result, sdpa.attributes.get_k());
            appendMatchKeyTensor(result, sdpa.attributes.get_v());
            appendMatchKeyTensor(result, sdpa.attributes.get_o());
            appendMatchKeyTensor(result, sdpa.attributes.get_do());
            appendMatchKeyTensor(result, sdpa.attributes.get_stats());
            handled = true;
            break;
        }
#endif
        // CANONICAL-TENSOR-ORDER: pointwise match key is the
        // pointwise_attributes.fbs INPUT-field declaration order with the output
        // field (out_0) dropped; the match is POSITIONAL. The required-input
        // count is mode-dependent, so the op string and tensor set are BOTH
        // derived from the SAME arity classifier
        // (hipdnn_frontend::isUnaryPointwiseMode / isBinaryPointwiseMode) used by
        // getCoreOperationName above, keeping the string and the emitted set in
        // lockstep:
        //   - pointwise_unary  → (in_0)
        //   - pointwise_binary → (in_0, in_1)
        // Ternary is intentionally not handled (no factory/functor), so a ternary
        // pointwise node produces an empty match key and is unsupported for
        // config round-trip. This MUST stay in lockstep with the backend reader
        // that consumes these tensors: the pointwise dispatch in
        // hipdnn_backend::heuristics::config::matchOverrideConfig in
        // backend/src/heuristics/config/ConfigBuiltIn.cpp. Any change to a
        // per-arity set/order here MUST be mirrored there and vice versa.
        case graph::NodeType::POINTWISE:
        {
            const auto& pw = static_cast<const graph::PointwiseNode&>(node);
            const auto mode = pw.attributes.get_mode();
            if(isUnaryPointwiseMode(mode))
            {
                appendMatchKeyTensor(result, pw.attributes.get_input_0());
                handled = true;
            }
            else if(isBinaryPointwiseMode(mode))
            {
                appendMatchKeyTensor(result, pw.attributes.get_input_0());
                appendMatchKeyTensor(result, pw.attributes.get_input_1());
                handled = true;
            }
            break;
        }
        default:
            break;
        }
    });

    // An op with no op-aware branch above leaves `handled` false and `result`
    // empty: an unsupported op produces an EMPTY match key (it never matches a
    // backend config rule).
    return result;
}

} // namespace hipdnn_frontend::detail
