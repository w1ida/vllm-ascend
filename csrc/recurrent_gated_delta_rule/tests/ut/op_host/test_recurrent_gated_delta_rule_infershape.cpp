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
 * \file test_recurrent_gated_delta_rule_infershape.cpp
 * \brief Infershape unit tests for RecurrentGatedDeltaRule.
 *        Verifies output shapes and dtype propagation for both BF16 and FP32 state inputs.
 */

#include <iostream>
#include <gtest/gtest.h>

#include "../../../../../tests/common/infer_shape_context_faker.h"
#include "../../../../../tests/common/infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RecurrentGatedDeltaRuleInfershapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RecurrentGatedDeltaRuleInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RecurrentGatedDeltaRuleInfershapeTest TearDown" << std::endl;
    }
};

// Test0: BF16 state input — output `out` is BF16, output `state` dtype is BF16 (propagated).
TEST_F(RecurrentGatedDeltaRuleInfershapeTest, Test0_BF16State_OutputShapeAndDtype)
{
    int t = 128;
    int nk = 4;
    int dk = 8;
    int nv = 128;
    int dv = 128;
    int sBlockNum = 128;
    int b = 64;

    gert::StorageShape queryShape       = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape         = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape       = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape        = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape       = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};
    gert::StorageShape seqLengthsShape  = {{b}, {b}};
    gert::StorageShape ssmStateIdxShape = {{t}, {t}};
    gert::StorageShape gShape           = {{t, nv}, {t, nv}};

    gert::InfershapeContextPara infershapeContextPara("RecurrentGatedDeltaRule",
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
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // out: BF16
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // state output: BF16 (propagated from input)
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        }
    );

    // out shape: [T, nv, dv]; state shape: [sBlockNum, nv, dv, dk]
    std::vector<std::vector<int64_t>> expectOutputShape = {{t, nv, dv}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test1: FP32 state input — output `state` dtype must be FP32 (dtype propagation).
//        output `out` remains BF16 regardless of state dtype.
TEST_F(RecurrentGatedDeltaRuleInfershapeTest, Test1_FP32State_DtypePropagation)
{
    int t = 128;
    int nk = 4;
    int dk = 8;
    int nv = 128;
    int dv = 128;
    int sBlockNum = 128;
    int b = 64;

    gert::StorageShape queryShape       = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape         = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape       = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape        = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape       = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};
    gert::StorageShape seqLengthsShape  = {{b}, {b}};
    gert::StorageShape ssmStateIdxShape = {{t}, {t}};
    gert::StorageShape gShape           = {{t, nv}, {t, nv}};

    gert::InfershapeContextPara infershapeContextPara("RecurrentGatedDeltaRule",
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
            {{{}, {}}, ge::DT_BF16,  ge::FORMAT_ND},  // out: always BF16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // state output: FP32 (propagated from input)
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        }
    );

    // out shape: [T, nv, dv]; state dtype propagation verified via expected output desc
    std::vector<std::vector<int64_t>> expectOutputShape = {{t, nv, dv}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
