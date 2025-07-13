/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#pragma once
#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "hailo/hailort.hpp"
#include "hailo_execution_provider.h"

namespace onnxruntime {

using hailort::VDevice;
using hailort::MemoryView;
using hailort::ConfiguredInferModel;
using hailort::InferModel;

class HailoKernel final : public OpKernel {
public:
    HailoKernel(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;
    virtual ~HailoKernel();

private:
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(HailoKernel);

    hailo_status infer(OpKernelContext* context) const;
    hailo_status update_output_params(ConstPointerContainer<std::vector<NodeArg*>> &output_nodes, const std::vector<int64_t> &format_order_params);
    hailo_status update_input_params(ConstPointerContainer<std::vector<NodeArg*>> &input_nodes, const std::vector<int64_t> &format_order_params);
    static std::chrono::milliseconds get_infer_timeout_from_env_var();

    std::shared_ptr<VDevice> m_vdevice;
    std::shared_ptr<InferModel> m_infer_model;
    // Using a ptr since we are using m_configured_infer_model inside infer() func which is const (and some of ConfiguredInferModel funcs, like run(), are not const)
    std::unique_ptr<ConfiguredInferModel> m_configured_infer_model;
    std::vector<std::string> m_sorted_outputs_names;
    std::vector<std::string> m_sorted_inputs_names;

    // TODO: HRT-5221 Support NCHW transformations
    // Transforming the data from/to Hailo default format order (transformation from other format order implemented at Hailo to NHWC)
    std::vector<bool> m_input_should_double_order_conversion;
    std::vector<bool> m_output_should_double_order_conversion;

    std::chrono::milliseconds m_infer_timeout;
};

}  // namespace onnxruntime