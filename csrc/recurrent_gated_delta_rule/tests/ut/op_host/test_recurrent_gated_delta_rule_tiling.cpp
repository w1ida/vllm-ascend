/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_recurrent_gated_delta_rule_tiling.cpp
 * \brief Unit tests for RecurrentGatedDeltaRule tiling — covers both BF16 and FP32 ssm_state.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../../op_host/recurrent_gated_delta_rule_tiling.h"
#include "../../../../../tests/common/tiling_context_faker.h"
#include "../../../../../tests/common/tiling_case_executor.h"
#include "../../../../op_kernel/recurrent_gated_delta_rule_tiling_data.h"

using namespace std;
using namespace ge;
using namespace optiling;

class RecurrentGatedDeltaRuleTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RecurrentGatedDeltaRuleTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RecurrentGatedDeltaRuleTilingTest TearDown" << std::endl;
    }
};

// Test0: BF16 state — baseline case; stateIsFp32 must be 0.
TEST_F(RecurrentGatedDeltaRuleTilingTest, Test0_BF16State)
{
    optiling::RecurrentGatedDeltaRuleCompileInfo compileinfo = {40, 196608}; // aivNum, ubSize

    int t = 128;
    int nk = 4;
    int dk = 8;
    int nv = 128;
    int dv = 128;
    int sBlockNum = 128;
    int b = 64;

    gert::StorageShape queryShape          = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape            = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape          = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape           = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape          = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};
    gert::StorageShape seqLengthsShape     = {{b}, {b}};
    gert::StorageShape ssmStateIdxShape    = {{t}, {t}};
    gert::StorageShape gShape              = {{t, nv}, {t, nv}};
    gert::StorageShape outShape            = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape stateOutShape       = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};

    gert::TilingContextPara tilingContextPara("RecurrentGatedDeltaRule",
        {
            {queryShape,       ge::DT_BF16,  ge::FORMAT_ND},
            {keyShape,         ge::DT_BF16,  ge::FORMAT_ND},
            {valueShape,       ge::DT_BF16,  ge::FORMAT_ND},
            {betaShape,        ge::DT_BF16,  ge::FORMAT_ND},
            {stateShape,       ge::DT_BF16,  ge::FORMAT_ND},  // BF16 state
            {seqLengthsShape,  ge::DT_INT32, ge::FORMAT_ND},
            {ssmStateIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {gShape,           ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape,      ge::DT_BF16, ge::FORMAT_ND},
            {stateOutShape, ge::DT_BF16, ge::FORMAT_ND},  // output state BF16
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(success);

    // Verify stateIsFp32 == 0 for BF16 state
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(RecurrentGatedDeltaRule::RecurrentGatedDeltaRuleTilingData));
    const auto* td = reinterpret_cast<const RecurrentGatedDeltaRule::RecurrentGatedDeltaRuleTilingData*>(
        tilingInfo.tilingData.get());
    EXPECT_EQ(td->stateIsFp32, 0u) << "Expected stateIsFp32=0 for BF16 state input";
}

// Test1: FP32 state — new path added in this port; stateIsFp32 must be 1.
TEST_F(RecurrentGatedDeltaRuleTilingTest, Test1_FP32State)
{
    optiling::RecurrentGatedDeltaRuleCompileInfo compileinfo = {40, 196608}; // aivNum, ubSize

    int t = 128;
    int nk = 4;
    int dk = 8;
    int nv = 128;
    int dv = 128;
    int sBlockNum = 128;
    int b = 64;

    gert::StorageShape queryShape          = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape            = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape          = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape           = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape          = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};
    gert::StorageShape seqLengthsShape     = {{b}, {b}};
    gert::StorageShape ssmStateIdxShape    = {{t}, {t}};
    gert::StorageShape gShape              = {{t, nv}, {t, nv}};
    gert::StorageShape outShape            = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape stateOutShape       = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};

    gert::TilingContextPara tilingContextPara("RecurrentGatedDeltaRule",
        {
            {queryShape,       ge::DT_BF16,  ge::FORMAT_ND},
            {keyShape,         ge::DT_BF16,  ge::FORMAT_ND},
            {valueShape,       ge::DT_BF16,  ge::FORMAT_ND},
            {betaShape,        ge::DT_BF16,  ge::FORMAT_ND},
            {stateShape,       ge::DT_FLOAT, ge::FORMAT_ND},  // FP32 state
            {seqLengthsShape,  ge::DT_INT32, ge::FORMAT_ND},
            {ssmStateIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {gShape,           ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape,      ge::DT_BF16,  ge::FORMAT_ND},
            {stateOutShape, ge::DT_FLOAT, ge::FORMAT_ND},  // output state FP32 (propagated)
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(success);

    // Verify stateIsFp32 == 1 for FP32 state
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(RecurrentGatedDeltaRule::RecurrentGatedDeltaRuleTilingData));
    const auto* td = reinterpret_cast<const RecurrentGatedDeltaRule::RecurrentGatedDeltaRuleTilingData*>(
        tilingInfo.tilingData.get());
    EXPECT_EQ(td->stateIsFp32, 1u) << "Expected stateIsFp32=1 for FP32 state input";

    // FP32 elements are 4 bytes vs 2 for BF16; vStep should be <= half the BF16 vStep.
    // (arch35 halves vStep_; arch32 has a dynamic UB check — both must fit)
    EXPECT_GT(td->vStep, 0u) << "vStep must be positive";
}

// Test2: Invalid state dtype (e.g. INT8) — tiling must reject it.
TEST_F(RecurrentGatedDeltaRuleTilingTest, Test2_InvalidStateDtype)
{
    optiling::RecurrentGatedDeltaRuleCompileInfo compileinfo = {40, 196608};

    int t = 128, nk = 4, dk = 8, nv = 128, dv = 128, sBlockNum = 128, b = 64;

    gert::StorageShape queryShape       = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape         = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape       = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape        = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape       = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};
    gert::StorageShape seqLengthsShape  = {{b}, {b}};
    gert::StorageShape ssmStateIdxShape = {{t}, {t}};
    gert::StorageShape gShape           = {{t, nv}, {t, nv}};
    gert::StorageShape outShape         = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape stateOutShape    = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};

    gert::TilingContextPara tilingContextPara("RecurrentGatedDeltaRule",
        {
            {queryShape,       ge::DT_BF16,  ge::FORMAT_ND},
            {keyShape,         ge::DT_BF16,  ge::FORMAT_ND},
            {valueShape,       ge::DT_BF16,  ge::FORMAT_ND},
            {betaShape,        ge::DT_BF16,  ge::FORMAT_ND},
            {stateShape,       ge::DT_INT8,  ge::FORMAT_ND},  // unsupported dtype
            {seqLengthsShape,  ge::DT_INT32, ge::FORMAT_ND},
            {ssmStateIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {gShape,           ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape,      ge::DT_BF16, ge::FORMAT_ND},
            {stateOutShape, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success) << "Tiling should fail for unsupported INT8 state dtype";
}
