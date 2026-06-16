// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU-free unit test for the autotune config-file match-key selectors
// (hipdnn_frontend::detail::getCoreOperationName / getMatchKeyTensors). It
// builds real convolution graphs through the shared SelectorUnitGraph helper
// (which delegates structure to FrontendGraphFactory, no handle, no device) and
// asserts the op string plus the canonical tensor set/order the writer emits
// (Gap B).
//
// The helper assigns UIDs to the graph's physical tensors (the selector filters
// on has_uid() and pre-build graphs have none) and reports them back keyed by
// role/name, so each test DISCOVERS its tensors (x/w/dy) and forms its EXPECT
// values from the discovered UIDs rather than hand-rolling the graph.

#include <gtest/gtest.h>
#include <hipdnn_frontend/detail/GraphMatchKey.hpp>
#include <hipdnn_test_sdk/utilities/SelectorUnitGraph.hpp>

using namespace hipdnn_frontend;
using hipdnn_test_sdk::utilities::OperationType;
using hipdnn_test_sdk::utilities::SelectorUnitGraph;

class TestGraphMatchKey : public ::testing::Test
{
};

TEST_F(TestGraphMatchKey, ConvFpropOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_FORWARD);
    const auto& x = unitGraph.byName("x");
    const auto& w = unitGraph.byName("w");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "conv_fprop");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 2u);
    // Canonical order: (x, w); output (y) excluded (virtual by default).
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), w->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), w->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), w->get_stride());
}

TEST_F(TestGraphMatchKey, ConvDgradOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_BACKWARD_DATA);
    const auto& dy = unitGraph.byName("dy");
    const auto& w = unitGraph.byName("w");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "conv_dgrad");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 2u);
    // Canonical order: (dy, w); output (dx) excluded.
    EXPECT_EQ(tensors[0]->get_uid(), dy->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), w->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), dy->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), dy->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), w->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), w->get_stride());
}

TEST_F(TestGraphMatchKey, ConvWgradOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_BACKWARD_WEIGHTS);
    const auto& x = unitGraph.byName("x");
    const auto& dy = unitGraph.byName("dy");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "conv_wgrad");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 2u);
    // Canonical order: (x, dy) — note this is NOT the input enum order (dy, x);
    // output (dw) excluded.
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), dy->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), dy->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), dy->get_stride());
}

TEST_F(TestGraphMatchKey, MatmulOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::MATMUL);
    // Matmul factory tensor names are uppercase "A"/"B" (byName throws on an
    // unknown name, so a lowercase typo would fail the test, not pass vacuously).
    const auto& a = unitGraph.byName("A");
    const auto& b = unitGraph.byName("B");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "matmul");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 2u);
    // Canonical order: (a, b); output (c) excluded.
    EXPECT_EQ(tensors[0]->get_uid(), a->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), b->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), a->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), a->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), b->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), b->get_stride());
}

TEST_F(TestGraphMatchKey, RmsnormOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::RMSNORM);
    // RMSNorm factory tensor names are lowercase "x"/"scale"/"epsilon" (unlike
    // matmul A/B or sdpa Q/K/V); byName throws on an unknown name, so a
    // case typo would fail the test, not pass vacuously. epsilon is a real
    // UID-bearing value-tensor (dims {1}), so it participates in the match key.
    const auto& x = unitGraph.byName("x");
    const auto& scale = unitGraph.byName("scale");
    const auto& epsilon = unitGraph.byName("epsilon");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "rmsnorm");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 3u);
    // Canonical order: (x, scale, epsilon); output (y) and optional
    // bias/inv_rms excluded.
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), scale->get_uid());
    EXPECT_EQ(tensors[2]->get_uid(), epsilon->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), scale->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), scale->get_stride());
    EXPECT_EQ(tensors[2]->get_dim(), epsilon->get_dim());
    EXPECT_EQ(tensors[2]->get_stride(), epsilon->get_stride());
}

TEST_F(TestGraphMatchKey, LayernormOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::LAYERNORM);
    // Layernorm factory tensor names are lowercase "x"/"scale"/"bias"/"epsilon";
    // byName throws on an unknown name, so a case typo would fail the test, not
    // pass vacuously. epsilon is a real UID-bearing value-tensor, so it
    // participates in the match key.
    const auto& x = unitGraph.byName("x");
    const auto& scale = unitGraph.byName("scale");
    const auto& bias = unitGraph.byName("bias");
    const auto& epsilon = unitGraph.byName("epsilon");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "layernorm");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 4u);
    // Canonical order: (x, scale, bias, epsilon); outputs (y, mean,
    // inv_variance) excluded.
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), scale->get_uid());
    EXPECT_EQ(tensors[2]->get_uid(), bias->get_uid());
    EXPECT_EQ(tensors[3]->get_uid(), epsilon->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), scale->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), scale->get_stride());
    EXPECT_EQ(tensors[2]->get_dim(), bias->get_dim());
    EXPECT_EQ(tensors[2]->get_stride(), bias->get_stride());
    // epsilon is a pass-by-value scalar whose dims are only populated by
    // build()/validate(); the GPU-free SelectorUnitGraph never runs those, so
    // its dims/strides stay empty here. Assert UID identity only (above), not
    // dims/strides, for epsilon.
}

TEST_F(TestGraphMatchKey, BatchnormInferenceOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::BATCHNORM_INFERENCE);
    // Batchnorm-inference factory tensor names are lowercase except the
    // camelCase "invVariance"; byName throws on an unknown name, so a case typo
    // would fail the test, not pass vacuously. All five inputs are real
    // UID-bearing physical buffers (no optionals).
    const auto& x = unitGraph.byName("x");
    const auto& mean = unitGraph.byName("mean");
    const auto& invVariance = unitGraph.byName("invVariance");
    const auto& scale = unitGraph.byName("scale");
    const auto& bias = unitGraph.byName("bias");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "batchnorm_inference");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 5u);
    // Canonical order: (x, mean, inv_variance, scale, bias); output (y) excluded.
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), mean->get_uid());
    EXPECT_EQ(tensors[2]->get_uid(), invVariance->get_uid());
    EXPECT_EQ(tensors[3]->get_uid(), scale->get_uid());
    EXPECT_EQ(tensors[4]->get_uid(), bias->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), mean->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), mean->get_stride());
    EXPECT_EQ(tensors[2]->get_dim(), invVariance->get_dim());
    EXPECT_EQ(tensors[2]->get_stride(), invVariance->get_stride());
    EXPECT_EQ(tensors[3]->get_dim(), scale->get_dim());
    EXPECT_EQ(tensors[3]->get_stride(), scale->get_stride());
    EXPECT_EQ(tensors[4]->get_dim(), bias->get_dim());
    EXPECT_EQ(tensors[4]->get_stride(), bias->get_stride());
}

TEST_F(TestGraphMatchKey, BatchnormTrainingOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::BATCHNORM_TRAINING);
    // Batchnorm-training factory tensor names are lowercase
    // "x"/"scale"/"bias"/"epsilon"; byName throws on an unknown name, so a case
    // typo would fail the test, not pass vacuously. epsilon is a real
    // UID-bearing value-tensor, so it participates in the match key.
    const auto& x = unitGraph.byName("x");
    const auto& scale = unitGraph.byName("scale");
    const auto& bias = unitGraph.byName("bias");
    const auto& epsilon = unitGraph.byName("epsilon");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "batchnorm_training");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 4u);
    // Canonical order: (x, scale, bias, epsilon); outputs (y, mean,
    // inv_variance) and nullable running-stats excluded.
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), scale->get_uid());
    EXPECT_EQ(tensors[2]->get_uid(), bias->get_uid());
    EXPECT_EQ(tensors[3]->get_uid(), epsilon->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), scale->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), scale->get_stride());
    EXPECT_EQ(tensors[2]->get_dim(), bias->get_dim());
    EXPECT_EQ(tensors[2]->get_stride(), bias->get_stride());
    // epsilon is a pass-by-value scalar whose dims are only populated by
    // build()/validate(); the GPU-free SelectorUnitGraph never runs those, so its
    // dims/strides stay empty here. Assert UID identity only (above), not
    // dims/strides, for epsilon (mirrors the layernorm/rmsnorm epsilon caveat).
}

TEST_F(TestGraphMatchKey, BatchnormBackwardOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::BATCHNORM_BACKWARD);
    // Batchnorm-backward factory tensor names are lowercase "dy"/"x"/"scale";
    // byName throws on an unknown name, so a case typo would fail the test, not
    // pass vacuously. The optional mean/inv_variance inputs are built absent and
    // excluded from the key.
    const auto& dy = unitGraph.byName("dy");
    const auto& x = unitGraph.byName("x");
    const auto& scale = unitGraph.byName("scale");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "batchnorm_backward");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 3u);
    // Canonical order: (dy, x, scale); output gradients (dx, dscale, dbias) and
    // the optional mean/inv_variance inputs excluded.
    EXPECT_EQ(tensors[0]->get_uid(), dy->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[2]->get_uid(), scale->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), dy->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), dy->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[2]->get_dim(), scale->get_dim());
    EXPECT_EQ(tensors[2]->get_stride(), scale->get_stride());
}

TEST_F(TestGraphMatchKey, PointwiseUnaryOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::POINTWISE_UNARY);
    // Pointwise-unary factory tensor name is lowercase "x" (mode RELU_FWD);
    // byName throws on an unknown name, so a case typo would fail the test, not
    // pass vacuously. Arity (1 input) is derived from the mode, so the op string
    // is "pointwise_unary" and the key is (in_0) only.
    const auto& x = unitGraph.byName("x");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "pointwise_unary");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 1u);
    // Canonical order: (in_0); output (out_0) excluded.
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
}

TEST_F(TestGraphMatchKey, PointwiseBinaryOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::POINTWISE_BINARY);
    // Pointwise-binary factory tensor names are lowercase "x"/"y" (mode ADD);
    // byName throws on an unknown name, so a case typo would fail the test, not
    // pass vacuously. Arity (2 inputs) is derived from the mode, so the op string
    // is "pointwise_binary" and the key is (in_0, in_1).
    const auto& x = unitGraph.byName("x");
    const auto& y = unitGraph.byName("y");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "pointwise_binary");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 2u);
    // Canonical order: (in_0, in_1); output (out_0) excluded.
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), y->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), y->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), y->get_stride());
}

TEST_F(TestGraphMatchKey, UnsupportedOpYieldsEmptyMatchKeyAndGraphNameOpString)
{
    // REDUCTION has a factory builder (so it is GPU-free constructible) but NO
    // op-aware branch in getMatchKeyTensors and NO op-string case in
    // getCoreOperationName. It is intentionally unsupported for config
    // round-trip, so the selectors must report it as such: an EMPTY match key
    // and the graph-name op-string fallback (NOT a specific op string). This
    // pins the empty-key contract — a future regression (re-adding a generic
    // UID-sort fallback, or REDUCTION accidentally gaining a branch) is caught.
    SelectorUnitGraph unitGraph(OperationType::REDUCTION);

    // Graph-name fallback, not a specific op string (createReductionGraph sets
    // the graph name to "Test_Reduction").
    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "Test_Reduction");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    EXPECT_EQ(tensors.size(), 0u);
}

#ifdef HIPDNN_ENABLE_SDPA
TEST_F(TestGraphMatchKey, SdpaFwdOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::SDPA_FORWARD);
    // SDPA-forward factory tensor names are uppercase "Q"/"K"/"V" (byName throws
    // on an unknown name, so a lowercase typo would fail the test, not pass
    // vacuously).
    const auto& q = unitGraph.byName("Q");
    const auto& k = unitGraph.byName("K");
    const auto& v = unitGraph.byName("V");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "sdpa_fwd");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 3u);
    // Canonical order: (q, k, v); output (o) and all optional inputs excluded.
    EXPECT_EQ(tensors[0]->get_uid(), q->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), k->get_uid());
    EXPECT_EQ(tensors[2]->get_uid(), v->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), q->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), q->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), k->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), k->get_stride());
    EXPECT_EQ(tensors[2]->get_dim(), v->get_dim());
    EXPECT_EQ(tensors[2]->get_stride(), v->get_stride());
}

TEST_F(TestGraphMatchKey, SdpaBwdOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::SDPA_BACKWARD);
    // SDPA-backward factory tensor names are "Q"/"K"/"V"/"O"/"dO"/"stats"
    // (case-sensitive: "dO" mixed-case, "stats" lowercase). byName throws on an
    // unknown name, so a typo would fail the test, not pass vacuously.
    const auto& q = unitGraph.byName("Q");
    const auto& k = unitGraph.byName("K");
    const auto& v = unitGraph.byName("V");
    const auto& o = unitGraph.byName("O");
    const auto& dO = unitGraph.byName("dO");
    const auto& stats = unitGraph.byName("stats");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "sdpa_bwd");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 6u);
    // Canonical order: (q, k, v, o, dO, stats); output gradients (dq, dk, dv)
    // and all optional inputs excluded.
    EXPECT_EQ(tensors[0]->get_uid(), q->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), k->get_uid());
    EXPECT_EQ(tensors[2]->get_uid(), v->get_uid());
    EXPECT_EQ(tensors[3]->get_uid(), o->get_uid());
    EXPECT_EQ(tensors[4]->get_uid(), dO->get_uid());
    EXPECT_EQ(tensors[5]->get_uid(), stats->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), q->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), q->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), k->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), k->get_stride());
    EXPECT_EQ(tensors[2]->get_dim(), v->get_dim());
    EXPECT_EQ(tensors[2]->get_stride(), v->get_stride());
    EXPECT_EQ(tensors[3]->get_dim(), o->get_dim());
    EXPECT_EQ(tensors[3]->get_stride(), o->get_stride());
    EXPECT_EQ(tensors[4]->get_dim(), dO->get_dim());
    EXPECT_EQ(tensors[4]->get_stride(), dO->get_stride());
    EXPECT_EQ(tensors[5]->get_dim(), stats->get_dim());
    EXPECT_EQ(tensors[5]->get_stride(), stats->get_stride());
}
#endif // HIPDNN_ENABLE_SDPA
