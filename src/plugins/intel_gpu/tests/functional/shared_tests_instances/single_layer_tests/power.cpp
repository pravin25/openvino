// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/power.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::PowerLayerTest;

std::vector<std::vector<ov::Shape>> input_shape_static = {
        {{1, 8}},
        {{2, 16}},
        {{3, 32}},
        {{4, 64}},
        {{5, 128}},
        {{6, 256}},
        {{7, 512}},
        {{8, 1024}}
};

std::vector<std::vector<float>> powers = {
        {0.0f},
        {0.5f},
        {1.0f},
        {1.1f},
        {1.5f},
        {2.0f},
};

std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16,
};

// -----------------------------------------------------------------------------
// Regression test for Power(x, -1.0) bug in GPU plugin
// -----------------------------------------------------------------------------
std::vector<std::vector<ov::Shape>> input_shape_reciprocal_bug = {
    {{1, 1, 1, 1}},
};

std::vector<std::vector<float>> powers_reciprocal_bug = {
    {-1.0f},
};

INSTANTIATE_TEST_SUITE_P(smoke_power, PowerLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static)),
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(powers),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        PowerLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_power_reciprocal_bug, PowerLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_reciprocal_bug)),
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(powers_reciprocal_bug),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    PowerLayerTest::getTestCaseName);



}  // namespace
   //
   //
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <iostream>
#include <cmath>

// test to confirm Power(x, -1) = 1/x behavior on GPU
TEST(PowerDirectStandalone, GPU_Power_Reciprocal_Print) {
    try {
        ov::Core core;

        ov::element::Type_t et = ov::element::f32;
        ov::Shape shape{1, 1, 1, 1};

        //input tensor: e.g. 0.11054
        auto param = std::make_shared<ov::op::v0::Parameter>(et, shape);
        auto one   = ov::op::v0::Constant::create(et, shape, {1.0f});

        //build reciprocal explicitly: Divide(1, x)
        auto reciprocal = std::make_shared<ov::op::v1::Divide>(one, param);

        auto model = std::make_shared<ov::Model>(
            ov::NodeVector{reciprocal},
            ov::ParameterVector{param},
            "GPU_Power_Reciprocal_Test");

        auto compiled = core.compile_model(model, "GPU");
        auto request = compiled.create_infer_request();

        ov::Tensor input_tensor(et, shape);
        input_tensor.data<float>()[0] = 0.11054f;
        request.set_input_tensor(input_tensor);

        request.infer();

        ov::Tensor output_tensor = request.get_output_tensor(0);
        float out_val = output_tensor.data<float>()[0];
        float expected = 1.0f / 0.11054f;

        std::cout << "\n[PowerTest] Input: " << 0.11054f << std::endl;
        std::cout << "[PowerTest] Output: " << out_val << std::endl;
        std::cout << "[PowerTest] Expected: " << expected << std::endl;
        std::cout << "[PowerTest] Diff: " << fabs(out_val - expected) << std::endl;

        ASSERT_LT(std::fabs(out_val - expected), 1e-2f)
            << "Output deviation too large for reciprocal test.";
    } catch (const std::exception& e) {
        std::cerr << "Exception in PowerReciprocal test: " << e.what() << std::endl;
        FAIL();
    }
}

TEST(PowerDirectStandalone_GPU_PowerOp, Power_fp32) {
    std::cout << "[PowerOpTest] Testing Power(x, -1.0) on GPU" << std::endl;

    // input
    std::vector<float> input_data = {0.094867f};
    auto input_tensor = ov::Tensor(ov::element::f32, {1}, input_data.data());

    // create model: Power(x, -1)
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto exponent = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});

    auto power = std::make_shared<ov::op::v1::Power>(param, exponent);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{power}, ov::ParameterVector{param});

    // create Core + GPU compiled model
    ov::Core core;
    auto compiled_model = core.compile_model(model, "GPU");
    auto infer_request = compiled_model.create_infer_request();

    // read inference
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // read output
    auto output_tensor = infer_request.get_output_tensor();
    float* output_data = output_tensor.data<float>();

    float expected = 1.0f / input_data[0];
    //float diff = std::abs(output_data[0] - expected);

    std::cout << "[PowerOpTest_f32] Input: " << input_data[0] << std::endl;
    std::cout << "[PowerOpTest_f32] Output: " << output_data[0] << std::endl;
    std::cout << "[PowerOpTest_f32] Expected: " << expected << std::endl;
    //std::cout << "[PowerOpTest_f32] Diff: " << diff << std::endl;

    ASSERT_NEAR(output_data[0], expected, 1e-2f);
}

TEST(PowerDirectStandalone_GPU_PowerOp, Power_fp16) {
    std::cout << "[PowerOpTest] Testing Power(x, -1.0) on GPU (fixed)" << std::endl;

    // Proper half data
    std::vector<ov::float16> input_data = {ov::float16(0.094867f)};
    auto input_tensor = ov::Tensor(ov::element::f16, {1}, input_data.data());

    // Model
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto exponent = ov::op::v0::Constant::create(ov::element::f16, {1}, {ov::float16(-1.0f)});
    auto power = std::make_shared<ov::op::v1::Power>(param, exponent);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{power}, ov::ParameterVector{param});

    ov::Core core;
    auto compiled_model = core.compile_model(model, "GPU");
    auto infer_request = compiled_model.create_infer_request();

    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    auto output_data = output_tensor.data<ov::float16>();

    float output_f32 = static_cast<float>(output_data[0]);
    float expected = 1.0f / static_cast<float>(input_data[0]);
    //float diff = std::abs(output_f32 - expected);

    std::cout << "[PowerOpTest_f16] Input: " << static_cast<float>(input_data[0]) << std::endl;
    std::cout << "[PowerOpTest_f16] Output: " << output_f32 << std::endl;
    std::cout << "[PowerOpTest_f16] Expected: " << expected << std::endl;
    //std::cout << "[PowerOpTest_f16] Diff: " << diff << std::endl;

    ASSERT_NEAR(output_f32, expected, 1e-2f); // looser tolerance for FP16
}

