// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file ConfigBuiltIn.cpp
 * @brief Backend-internal implementation of the
 *        SelectionHeuristic::Config policy.
 *
 * The policy reads HIPDNN_HEUR_CONFIG_PATH (a JSON file mapping
 * conv-shape patterns to engine names via EngineOverrideConfig), walks
 * conv-like nodes in the serialized graph, and on the first matching rule
 * reorders the candidate engine IDs so the chosen engine is first. When the
 * env var is unset, the file is missing/invalid, no rule matches, or the
 * matched engine is not among the candidates, the policy declines so the
 * outer policy loop can try the next plugin.
 *
 * Mechanics mirror StaticOrderingBuiltIn — a function-pointer table wrapped
 * by HeuristicPlugin::createBuiltIn so registration and validation flow
 * through the same paths as dlopen-loaded plugins.
 */

#include "ConfigBuiltIn.hpp"

#include "EngineOverrideConfig.hpp"
#include "heuristics/BuiltInLogging.hpp"
#include "logging/Logging.hpp"

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_plugin_sdk/HeuristicValidation.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_backend::heuristics::config
{
namespace
{

constexpr const char* PLUGIN_NAME = "BuiltInConfigHeuristic";
constexpr const char* PLUGIN_VERSION = "1.0.0";
constexpr const char* POLICY_NAME = "SelectionHeuristic::Config";

// File-scope logging callback / level, set via the C-ABI-shaped
// SetLoggingCallback / SetLogLevel below. The backend supplies its own
// callback when registering the built-in (see PluginManagerBase::registerPlugin)
// so log lines from this module flow through the backend logger.
//
// Identity contract: the built-in is statically linked into the backend, so
// these globals live in the same process image as the caller. The last writer
// wins — if multiple HeuristicPluginManager instances register the built-in
// they overwrite each other's callback. This is intentional: registerPlugin()
// hands in a callback that forwards to the backend logger, which is itself a
// process-wide sink, so the identity of the "current" callback does not matter
// as long as one is installed. Do not assume per-instance scoping here.
hipdnnCallback_t g_loggingCallback = nullptr; // NOLINT(readability-identifier-naming)
hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO; // NOLINT(readability-identifier-naming)

#define CONFIG_BUILTIN_LOG(severity, ...) \
    HIPDNN_BUILTIN_HEURISTIC_LOG(         \
        g_loggingCallback, g_logLevel, severity, "[BuiltInConfig] ", __VA_ARGS__)

int64_t policyId()
{
    static const int64_t s_id = hipdnn_data_sdk::utilities::policyNameToId(POLICY_NAME);
    return s_id;
}

// ---- Graph-walk helpers (lifted from the former PreferredEngineResolver) ---

using hipdnn_flatbuffers_sdk::data_objects::Graph;

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>* fb)
{
    if(fb == nullptr)
    {
        return {};
    }
    return {fb->begin(), fb->end()};
}

struct TensorDimsStrides
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

std::unordered_map<int64_t, TensorDimsStrides> indexTensorsByUid(const Graph* graph)
{
    std::unordered_map<int64_t, TensorDimsStrides> out;
    const auto* tensors = graph->tensors();
    if(tensors == nullptr)
    {
        return out;
    }
    out.reserve(tensors->size());
    for(const auto* t : *tensors)
    {
        if(t == nullptr)
        {
            continue;
        }
        out.emplace(t->uid(), TensorDimsStrides{toVector(t->dims()), toVector(t->strides())});
    }
    return out;
}

std::optional<int64_t>
    matchOverrideConfig(const EngineOverrideConfig& config,
                        const Graph* graph,
                        const std::unordered_map<int64_t, TensorDimsStrides>& tensorIndex)
{
    const auto* nodes = graph->nodes();
    if(nodes == nullptr)
    {
        return std::nullopt;
    }

    auto viewFor = [&](int64_t uid) -> const TensorDimsStrides* {
        auto it = tensorIndex.find(uid);
        return it == tensorIndex.end() ? nullptr : &it->second;
    };

    auto buildView = [&](const TensorDimsStrides* t) { return TensorView{&t->dims, &t->strides}; };

    for(const auto* node : *nodes)
    {
        if(node == nullptr)
        {
            continue;
        }

        const char* op = nullptr;
        // Per-op canonical input tensors in match-key order. Each op branch fills
        // this accumulator with its required-input views (in declaration order);
        // the guard below requires the FIRST required input to be present and
        // drops any null trailing tensor. Every multi-input op (conv/matmul=2,
        // sdpa_fwd/rmsnorm=3, layernorm=4, sdpa_bwd=6, batchnorm 3-5) supplies
        // non-null leading inputs, so its emitted view vector is unchanged;
        // pointwise_unary is the first 1-input op the relaxed guard admits.
        std::vector<const TensorDimsStrides*> opTensors;

        // CANONICAL-TENSOR-ORDER: the per-op tensor order supplied here is the
        // op's flatbuffer `*_attributes.fbs` INPUT-field declaration order with
        // the output field(s) dropped; the comparison is POSITIONAL (the views
        // below are matched index-by-index against the config rule). This MUST
        // stay in lockstep with the frontend writer/selector that produces these
        // tensors: `hipdnn_frontend::detail::getMatchKeyTensors` in
        // `frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp`. Any change
        // to a per-op set/order here MUST be mirrored there and vice versa.
        // Convolution (output excluded, 2 inputs each): conv_fprop → (x, w);
        // conv_dgrad → (dy, w); conv_wgrad → (x, dy).
        if(const auto* fwd = node->attributes_as_ConvolutionFwdAttributes())
        {
            op = "conv_fprop";
            opTensors = {viewFor(fwd->x_tensor_uid()), viewFor(fwd->w_tensor_uid())};
        }
        else if(const auto* bwd = node->attributes_as_ConvolutionBwdAttributes())
        {
            op = "conv_dgrad";
            opTensors = {viewFor(bwd->dy_tensor_uid()), viewFor(bwd->w_tensor_uid())};
        }
        else if(const auto* wrw = node->attributes_as_ConvolutionWrwAttributes())
        {
            op = "conv_wgrad";
            opTensors = {viewFor(wrw->x_tensor_uid()), viewFor(wrw->dy_tensor_uid())};
        }
        // CANONICAL-TENSOR-ORDER: matmul tensor order is the
        // matmul_attributes.fbs INPUT-field declaration order (a, b); the output
        // field c is dropped; the comparison is POSITIONAL. This mirrors the
        // frontend writer/selector hipdnn_frontend::detail::getMatchKeyTensors in
        // frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp. Any change to
        // this set/order MUST be mirrored there and vice versa.
        else if(const auto* mm = node->attributes_as_MatmulAttributes())
        {
            op = "matmul";
            opTensors = {viewFor(mm->a_tensor_uid()), viewFor(mm->b_tensor_uid())};
        }
        // CANONICAL-TENSOR-ORDER: rmsnorm tensor order is the
        // rmsnorm_attributes.fbs INPUT-field declaration order (x, scale,
        // epsilon); the output field y and the optional bias/inv_rms fields are
        // dropped; the comparison is POSITIONAL. epsilon is a real UID-bearing
        // value-tensor (dims {1}), NOT a dropped scalar, so it is the third
        // positional input. This mirrors the frontend writer/selector
        // hipdnn_frontend::detail::getMatchKeyTensors in
        // frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp. Any change to
        // this set/order MUST be mirrored there and vice versa.
        else if(const auto* rms = node->attributes_as_RMSNormAttributes())
        {
            op = "rmsnorm";
            opTensors = {viewFor(rms->x_tensor_uid()),
                         viewFor(rms->scale_tensor_uid()),
                         viewFor(rms->epsilon_tensor_uid())};
        }
        // CANONICAL-TENSOR-ORDER: layernorm tensor order is the
        // layernorm_attributes.fbs INPUT-field declaration order (x, scale, bias,
        // epsilon); the output fields y/mean/inv_variance and the scalar attrs
        // (normalized_dim_count, forward_phase) are dropped; the comparison is
        // POSITIONAL. epsilon is a real UID-bearing value-tensor (dims {1}), NOT a
        // dropped scalar, so it is the fourth positional input. This mirrors the
        // frontend writer/selector hipdnn_frontend::detail::getMatchKeyTensors in
        // frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp. Any change to
        // this set/order MUST be mirrored there and vice versa.
        else if(const auto* ln = node->attributes_as_LayernormAttributes())
        {
            op = "layernorm";
            opTensors = {viewFor(ln->x_tensor_uid()),
                         viewFor(ln->scale_tensor_uid()),
                         viewFor(ln->bias_tensor_uid()),
                         viewFor(ln->epsilon_tensor_uid())};
        }
        // CANONICAL-TENSOR-ORDER: batchnorm_inference tensor order is the
        // batchnorm_inference_attributes.fbs INPUT-field declaration order
        // (x, mean, inv_variance, scale, bias); the output field y is dropped;
        // the comparison is POSITIONAL. All five are real UID-bearing physical
        // buffers (no optionals), so it yields a 5-element view vector. This
        // mirrors the frontend writer/selector
        // hipdnn_frontend::detail::getMatchKeyTensors in
        // frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp. Any change to
        // this set/order MUST be mirrored there and vice versa.
        else if(const auto* bnInf = node->attributes_as_BatchnormInferenceAttributes())
        {
            op = "batchnorm_inference";
            opTensors = {viewFor(bnInf->x_tensor_uid()),
                         viewFor(bnInf->mean_tensor_uid()),
                         viewFor(bnInf->inv_variance_tensor_uid()),
                         viewFor(bnInf->scale_tensor_uid()),
                         viewFor(bnInf->bias_tensor_uid())};
        }
        // CANONICAL-TENSOR-ORDER: batchnorm_training tensor order is the
        // batchnorm_attributes.fbs INPUT-field declaration order
        // (x, scale, bias, epsilon); the output fields y/mean/inv_variance and
        // the nullable running-stats fields are dropped; the comparison is
        // POSITIONAL. epsilon is a real UID-bearing value-tensor (dims {1}), NOT
        // a dropped scalar, so it is the fourth positional input. This mirrors
        // the frontend writer/selector
        // hipdnn_frontend::detail::getMatchKeyTensors in
        // frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp. Any change to
        // this set/order MUST be mirrored there and vice versa.
        else if(const auto* bnTrain = node->attributes_as_BatchnormAttributes())
        {
            op = "batchnorm_training";
            opTensors = {viewFor(bnTrain->x_tensor_uid()),
                         viewFor(bnTrain->scale_tensor_uid()),
                         viewFor(bnTrain->bias_tensor_uid()),
                         viewFor(bnTrain->epsilon_tensor_uid())};
        }
        // CANONICAL-TENSOR-ORDER: batchnorm_backward tensor order is the
        // batchnorm_backward_attributes.fbs REQUIRED INPUT-field declaration
        // order (dy, x, scale); the output gradient fields (dx/dscale/dbias), the
        // nullable mean/inv_variance inputs, and the peer-stats vector are
        // dropped; the comparison is POSITIONAL. This mirrors the frontend
        // writer/selector hipdnn_frontend::detail::getMatchKeyTensors in
        // frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp. Any change to
        // this set/order MUST be mirrored there and vice versa.
        else if(const auto* bnBwd = node->attributes_as_BatchnormBackwardAttributes())
        {
            op = "batchnorm_backward";
            opTensors = {viewFor(bnBwd->dy_tensor_uid()),
                         viewFor(bnBwd->x_tensor_uid()),
                         viewFor(bnBwd->scale_tensor_uid())};
        }
#ifdef HIPDNN_ENABLE_SDPA
        // CANONICAL-TENSOR-ORDER: sdpa_fwd tensor order is the
        // sdpa_attributes.fbs REQUIRED INPUT fields (q, k, v); the output field o
        // and every optional input/output field are dropped; the comparison is
        // POSITIONAL. sdpa_fwd is a 3-input op, so it yields a 3-element view
        // vector. This mirrors the frontend writer/selector
        // hipdnn_frontend::detail::getMatchKeyTensors in
        // frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp. Any change to
        // this set/order MUST be mirrored there and vice versa.
        else if(const auto* sdpa = node->attributes_as_SdpaAttributes())
        {
            op = "sdpa_fwd";
            opTensors = {viewFor(sdpa->q_tensor_uid()),
                         viewFor(sdpa->k_tensor_uid()),
                         viewFor(sdpa->v_tensor_uid())};
        }
        // CANONICAL-TENSOR-ORDER: sdpa_bwd tensor order is the
        // sdpa_backward_attributes.fbs REQUIRED INPUT fields (q, k, v, o, dO,
        // stats) in declaration order; the output gradient fields (dq, dk, dv)
        // and every optional input/output field are dropped; the comparison is
        // POSITIONAL. sdpa_bwd is a 6-input op, so it yields a 6-element view
        // vector. This mirrors the frontend writer/selector
        // hipdnn_frontend::detail::getMatchKeyTensors in
        // frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp. Any change to
        // this set/order MUST be mirrored there and vice versa.
        else if(const auto* sdpaBwd = node->attributes_as_SdpaBackwardAttributes())
        {
            op = "sdpa_bwd";
            opTensors = {viewFor(sdpaBwd->q_tensor_uid()),
                         viewFor(sdpaBwd->k_tensor_uid()),
                         viewFor(sdpaBwd->v_tensor_uid()),
                         viewFor(sdpaBwd->o_tensor_uid()),
                         viewFor(sdpaBwd->do_tensor_uid()),
                         viewFor(sdpaBwd->stats_tensor_uid())};
        }
#endif
        // CANONICAL-TENSOR-ORDER: pointwise tensor order is the
        // pointwise_attributes.fbs INPUT-field declaration order with the output
        // field (out_0) dropped; the comparison is POSITIONAL. The required-input
        // count is mode-dependent, so the op string and tensor set are BOTH
        // derived from the SAME arity classifier
        // (hipdnn_flatbuffers_sdk::utilities::isUnaryPointwiseMode /
        // isBinaryPointwiseMode) the frontend mirrors with
        // hipdnn_frontend::is{Unary,Binary}PointwiseMode, so both sides agree on
        // arity by construction:
        //   - pointwise_unary  → (in_0)
        //   - pointwise_binary → (in_0, in_1)
        // Ternary is intentionally NOT dispatched here (no factory/functor), so a
        // ternary pointwise node is left unmatched. This mirrors the frontend
        // writer/selector hipdnn_frontend::detail::getMatchKeyTensors in
        // frontend/include/hipdnn_frontend/detail/GraphMatchKey.hpp. Any change to
        // a per-arity set/order here MUST be mirrored there and vice versa.
        else if(const auto* pw = node->attributes_as_PointwiseAttributes())
        {
            const auto mode = pw->operation();
            if(hipdnn_flatbuffers_sdk::utilities::isUnaryPointwiseMode(mode))
            {
                op = "pointwise_unary";
                opTensors = {viewFor(pw->in_0_tensor_uid())};
            }
            else if(hipdnn_flatbuffers_sdk::utilities::isBinaryPointwiseMode(mode))
            {
                op = "pointwise_binary";
                // in_1 is Optional<int64_t> in the flatbuffer but is always
                // present for a binary mode (the writer emits it); deref after the
                // classifier confirms binary arity.
                opTensors
                    = {viewFor(pw->in_0_tensor_uid()), viewFor(pw->in_1_tensor_uid().value())};
            }
        }

        // Require the op and at least its first required input present. Each op
        // pushes its own required-input count (pointwise_unary=1, conv/matmul=2,
        // sdpa_fwd=3, rmsnorm=3, layernorm=4, sdpa_bwd=6); a null leading tensor
        // or an empty accumulator means the node is malformed/unindexed → skip.
        if(op == nullptr || opTensors.empty() || opTensors[0] == nullptr)
        {
            continue;
        }

        // Build the positional view vector from every required tensor that
        // resolved; a null trailing tensor is dropped (matching prior behavior).
        std::vector<TensorView> views;
        views.reserve(opTensors.size());
        for(const auto* t : opTensors)
        {
            if(t != nullptr)
            {
                views.push_back(buildView(t));
            }
        }
        auto match = config.matchOperation(op, views);
        if(match.has_value())
        {
            return match;
        }
    }
    return std::nullopt;
}

/// Validate the buffer and return the typed Graph root, or nullptr on failure.
const Graph* parseGraphBuffer(const std::vector<uint8_t>& buffer)
{
    if(buffer.empty())
    {
        return nullptr;
    }
    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if(!verifier.VerifyBuffer<Graph>())
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_WARN, "policyFinalize: invalid serialized graph buffer");
        return nullptr;
    }
    return hipdnn_flatbuffers_sdk::data_objects::GetGraph(buffer.data());
}

/// Reorder @p candidates so @p preferredEngineId comes first, preserving the
/// relative order of the rest. Returns nullopt if the preferred id is not in
/// the candidate list.
std::optional<std::vector<int64_t>>
    reorderWithPreferredFirst(const std::vector<int64_t>& candidates, int64_t preferredEngineId)
{
    auto it = std::find(candidates.begin(), candidates.end(), preferredEngineId);
    if(it == candidates.end())
    {
        return std::nullopt;
    }
    std::vector<int64_t> reordered;
    reordered.reserve(candidates.size());
    reordered.push_back(preferredEngineId);
    for(const int64_t engineId : candidates)
    {
        if(engineId != preferredEngineId)
        {
            reordered.push_back(engineId);
        }
    }
    return reordered;
}

// ---- Per-handle / per-descriptor state -------------------------------------

struct Handle
{
    std::vector<uint8_t> devicePropertiesBuffer;
    bool devicePropertiesSet = false;
};

struct PolicyDescriptor
{
    Handle* handle = nullptr;
    std::vector<int64_t> candidateEngineIds;
    std::vector<uint8_t> serializedGraph;
    std::vector<int64_t> sortedEngineIds;
    bool finalized = false;

    explicit PolicyDescriptor(Handle* h)
        : handle(h)
    {
    }
};

// ---- Base plugin metadata --------------------------------------------------

hipdnnPluginStatus_t getName(const char** name)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(name, CONFIG_BUILTIN_LOG, "getName: null output pointer");
    *name = PLUGIN_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getVersion(const char** version)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(version, CONFIG_BUILTIN_LOG, "getVersion: null output pointer");
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getApiVersion(const char** version)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        version, CONFIG_BUILTIN_LOG, "getApiVersion: null output pointer");
    *version = HIPDNN_HEURISTIC_API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getType(hipdnnPluginType_t* type)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(type, CONFIG_BUILTIN_LOG, "getType: null output pointer");
    *type = HIPDNN_PLUGIN_TYPE_HEURISTIC;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t setLoggingCallback(hipdnnCallback_t callback)
{
    g_loggingCallback = callback;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t setLogLevel(hipdnnSeverity_t level)
{
    g_logLevel = level;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

void getLastErrorString(const char** errorStr)
{
    if(errorStr == nullptr)
    {
        return;
    }
    *errorStr = "No error information available";
}

// ---- Policy enumeration ----------------------------------------------------

hipdnnPluginStatus_t
    getAllPolicyIds(int64_t* policyIds, uint32_t maxPolicies, uint32_t* numPolicies)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numPolicies, CONFIG_BUILTIN_LOG, "getAllPolicyIds: null num_policies");

    constexpr uint32_t TOTAL_POLICIES = 1;
    *numPolicies = TOTAL_POLICIES;
    if(policyIds == nullptr || maxPolicies == 0)
    {
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    if(maxPolicies < TOTAL_POLICIES)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    policyIds[0] = policyId();
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getPolicyName(int64_t id, const char** name)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(name, CONFIG_BUILTIN_LOG, "getPolicyName: null output pointer");
    if(id != policyId())
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "getPolicyName: unknown policy ID");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = POLICY_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ---- Handle lifecycle ------------------------------------------------------

hipdnnPluginStatus_t handleCreate(hipdnnHeuristicHandle_t* outHandle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outHandle, CONFIG_BUILTIN_LOG, "handleCreate: null output pointer");
    try
    {
        auto h = std::make_unique<Handle>();
        *outHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(h.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "handleCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t handleDestroy(hipdnnHeuristicHandle_t handle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(handle, CONFIG_BUILTIN_LOG, "handleDestroy: null handle");
    delete reinterpret_cast<Handle*>(handle);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t handleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                               const hipdnnPluginConstData_t* devicePropsSerialized)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        handle, CONFIG_BUILTIN_LOG, "handleSetDeviceProperties: null handle");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(devicePropsSerialized,
                                     true,
                                     CONFIG_BUILTIN_LOG,
                                     "handleSetDeviceProperties: invalid buffer");
    try
    {
        auto* h = reinterpret_cast<Handle*>(handle);
        const auto* data = reinterpret_cast<const uint8_t*>(devicePropsSerialized->ptr);
        h->devicePropertiesBuffer.assign(data, data + devicePropsSerialized->size);
        h->devicePropertiesSet = true;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "handleSetDeviceProperties failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Policy descriptor lifecycle ------------------------------------------

hipdnnPluginStatus_t policyDescriptorCreate(hipdnnHeuristicHandle_t pluginHandle,
                                            int64_t id,
                                            hipdnnHeuristicPolicyDescriptor_t* outDesc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        pluginHandle, CONFIG_BUILTIN_LOG, "policyDescriptorCreate: null handle");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outDesc, CONFIG_BUILTIN_LOG, "policyDescriptorCreate: null output pointer");
    if(id != policyId())
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policyDescriptorCreate: unknown policy ID");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    try
    {
        auto desc = std::make_unique<PolicyDescriptor>(reinterpret_cast<Handle*>(pluginHandle));
        *outDesc = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(desc.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policyDescriptorCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, CONFIG_BUILTIN_LOG, "policyDescriptorDestroy: null descriptor");
    delete reinterpret_cast<PolicyDescriptor*>(desc);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ---- Policy inputs ---------------------------------------------------------

hipdnnPluginStatus_t policySetEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                        const int64_t* engineIds,
                                        size_t engineIdCount)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, CONFIG_BUILTIN_LOG, "policySetEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_ARRAY(engineIds,
                                engineIdCount,
                                CONFIG_BUILTIN_LOG,
                                "policySetEngineIds: null engine_ids with count > 0");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        d->candidateEngineIds.assign(engineIds, engineIds + engineIdCount);
        d->finalized = false;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policySetEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                              const hipdnnPluginConstData_t* serializedGraph)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, CONFIG_BUILTIN_LOG, "policySetSerializedGraph: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(serializedGraph,
                                     false,
                                     CONFIG_BUILTIN_LOG,
                                     "policySetSerializedGraph: invalid graph buffer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        const auto* bytes = reinterpret_cast<const uint8_t*>(serializedGraph->ptr);
        if(bytes == nullptr || serializedGraph->size == 0)
        {
            d->serializedGraph.clear();
        }
        else
        {
            d->serializedGraph.assign(bytes, bytes + serializedGraph->size);
        }
        d->finalized = false;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policySetSerializedGraph failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Selection -------------------------------------------------------------

hipdnnPluginStatus_t policyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* outApplied)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, CONFIG_BUILTIN_LOG, "policyFinalize: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outApplied, CONFIG_BUILTIN_LOG, "policyFinalize: null output pointer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        *outApplied = 0;

        if(d->candidateEngineIds.empty())
        {
            CONFIG_BUILTIN_LOG(HIPDNN_SEV_INFO, "policyFinalize: no candidate engines; declining");
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        const auto config = EngineOverrideConfig::loadFromEnv();
        if(!config.has_value())
        {
            CONFIG_BUILTIN_LOG(HIPDNN_SEV_INFO,
                               "policyFinalize: HIPDNN_HEUR_CONFIG_PATH unset or "
                               "unreadable; declining");
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        const Graph* graph = parseGraphBuffer(d->serializedGraph);
        if(graph == nullptr)
        {
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        const auto preferredEngineId
            = matchOverrideConfig(*config, graph, indexTensorsByUid(graph));
        if(!preferredEngineId.has_value())
        {
            CONFIG_BUILTIN_LOG(HIPDNN_SEV_INFO,
                               "policyFinalize: no rule matched any conv node; declining");
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        auto reordered = reorderWithPreferredFirst(d->candidateEngineIds, *preferredEngineId);
        if(!reordered.has_value())
        {
            CONFIG_BUILTIN_LOG(HIPDNN_SEV_INFO,
                               "policyFinalize: matched engine 0x%llx not in candidates; declining",
                               static_cast<unsigned long long>(*preferredEngineId));
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        CONFIG_BUILTIN_LOG(HIPDNN_SEV_INFO,
                           "policyFinalize: reordered %zu engines with preferred 0x%llx first",
                           reordered->size(),
                           static_cast<unsigned long long>(*preferredEngineId));
        d->sortedEngineIds = std::move(*reordered);
        d->finalized = true;
        *outApplied = 1;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policyFinalize failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                              int64_t* engineIds,
                                              size_t* numEngines)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, CONFIG_BUILTIN_LOG, "policyGetSortedEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numEngines, CONFIG_BUILTIN_LOG, "policyGetSortedEngineIds: null num_engines pointer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        if(!d->finalized)
        {
            CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR,
                               "policyGetSortedEngineIds: descriptor not finalized");
            return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
        }
        if(engineIds == nullptr)
        {
            *numEngines = d->sortedEngineIds.size();
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }
        *numEngines = std::min(*numEngines, d->sortedEngineIds.size());
        std::copy_n(d->sortedEngineIds.begin(), *numEngines, engineIds);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policyGetSortedEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

} // namespace

hipdnn_backend::plugin::HeuristicPluginFunctionTable populateFunctionTable()
{
    hipdnn_backend::plugin::HeuristicPluginFunctionTable funcs{};
    funcs.getName = &getName;
    funcs.getVersion = &getVersion;
    funcs.getApiVersion = &getApiVersion;
    funcs.getType = &getType;
    funcs.setLoggingCallback = &setLoggingCallback;
    funcs.setLogLevel = &setLogLevel;
    funcs.getLastErrorString = &getLastErrorString;
    funcs.getAllPolicyIds = &getAllPolicyIds;
    funcs.getPolicyName = &getPolicyName;
    funcs.handleCreate = &handleCreate;
    funcs.handleDestroy = &handleDestroy;
    funcs.handleSetDeviceProperties = &handleSetDeviceProperties;
    funcs.policyDescriptorCreate = &policyDescriptorCreate;
    funcs.policyDescriptorDestroy = &policyDescriptorDestroy;
    funcs.policySetEngineIds = &policySetEngineIds;
    funcs.policySetSerializedGraph = &policySetSerializedGraph;
    funcs.policyFinalize = &policyFinalize;
    funcs.policyGetSortedEngineIds = &policyGetSortedEngineIds;
    return funcs;
}

} // namespace hipdnn_backend::heuristics::config
