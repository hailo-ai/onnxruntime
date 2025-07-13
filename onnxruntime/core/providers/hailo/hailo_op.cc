/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "core/providers/shared_library/provider_api.h"
#include "hailo_op.h"
#include "utils.h"

#include <iostream>
#include <mutex>

namespace onnxruntime {

static constexpr const char* HEF_ATTRIBUTE = "hef";
static constexpr const char* SORTED_INPUT_NAMES_ATTRUBUTE = "sorted_input_names";
static constexpr const char* SORTED_OUTPUT_NAMES_ATTRUBUTE = "sorted_output_names";
static constexpr const char* INPUT_ORDER_ATTRIBUTE = "input_format_order";
static constexpr const char* OUTPUT_ORDER_ATTRIBUTE = "output_format_order";

#define HAILO_ONNXRT_INFER_TIMEOUT_MS_ENV_VAR ("HAILO_ONNXRT_INFER_TIMEOUT_MS")
constexpr auto INFER_DEFAULT_TIMEOUT = std::chrono::milliseconds(30000);

HailoKernel::HailoKernel(const OpKernelInfo& info) :
    OpKernel(info), m_infer_timeout(get_infer_timeout_from_env_var())
{
    std::string binary_hef;
    auto onnx_status = info.GetAttr(HEF_ATTRIBUTE, &binary_hef);
    HAILO_ORT_ENFORCE(onnx_status.IsOK(), "attribute '",  HEF_ATTRIBUTE, "' is not set");

    onnx_status = info.GetAttrs(SORTED_INPUT_NAMES_ATTRUBUTE, m_sorted_inputs_names);
    HAILO_ORT_ENFORCE(onnx_status.IsOK(), "attribute '",  SORTED_INPUT_NAMES_ATTRUBUTE, "' is not set");

    onnx_status = info.GetAttrs(SORTED_OUTPUT_NAMES_ATTRUBUTE, m_sorted_outputs_names);
    HAILO_ORT_ENFORCE(onnx_status.IsOK(), "attribute '",  SORTED_OUTPUT_NAMES_ATTRUBUTE, "' is not set");

    std::vector<int64_t> input_format_order;
    onnx_status = info.GetAttrs(INPUT_ORDER_ATTRIBUTE, input_format_order);
    HAILO_ORT_ENFORCE(onnx_status.IsOK(), "attribute '",  INPUT_ORDER_ATTRIBUTE, "' is not set");

    std::vector<int64_t> output_format_order;
    onnx_status = info.GetAttrs(OUTPUT_ORDER_ATTRIBUTE, output_format_order);
    HAILO_ORT_ENFORCE(onnx_status.IsOK(), "attribute '",  OUTPUT_ORDER_ATTRIBUTE, "' is not set");

    auto hef_memory_view = MemoryView::create_const(binary_hef.c_str(), binary_hef.length());

    hailo_vdevice_params_t params;
    hailo_status status = hailo_init_vdevice_params(&params);
    HAILO_ORT_ENFORCE(HAILO_SUCCESS == status, "Failed init vdevice_params, status = ", status);
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    params.group_id = "SHARED";
    auto expected_vdevice = VDevice::create(params);
    HAILO_CHECK_EXPECTED(expected_vdevice, "Failed to create VDevice");
    m_vdevice = std::move(expected_vdevice.value());

    auto infer_model_expected = m_vdevice->create_infer_model(hef_memory_view);
    HAILO_CHECK_EXPECTED(infer_model_expected, "Failed to create Infer Model");
    m_infer_model = infer_model_expected.release();

    auto input_nodes = info.node().InputDefs();
    status = update_input_params(input_nodes, input_format_order);
    HAILO_ORT_ENFORCE(HAILO_SUCCESS == status, "Failed to update input params");

    auto output_nodes = info.node().OutputDefs();
    status = update_output_params(output_nodes, output_format_order);
    HAILO_ORT_ENFORCE(HAILO_SUCCESS == status, "Failed to update output params");

    auto configured_infer_model_expected = m_infer_model->configure();
    HAILO_CHECK_EXPECTED(configured_infer_model_expected, "Failed to create Configured Infer Model");
    m_configured_infer_model = std::make_unique<ConfiguredInferModel>(configured_infer_model_expected.release());
}

HailoKernel::~HailoKernel()
{
}

hailo_status HailoKernel::update_output_params(ConstPointerContainer<std::vector<NodeArg*>> &output_nodes,
    const std::vector<int64_t> &format_order_params)
{
    HAILO_ORT_ENFORCE(output_nodes.size() == m_sorted_outputs_names.size(),
        "Number of output nodes = ", output_nodes.size(), ", is inconsistent with output vstreams number = ", m_sorted_outputs_names.size());

    for (size_t i = 0; i < m_sorted_outputs_names.size(); i++) {
        auto output_ort_dtype = output_nodes[i]->TypeAsProto()->tensor_type().elem_type();
        auto output_hailo_dtype = HailoUtils::convert_ort_to_hailo_dtype(output_ort_dtype);
        auto output_name = m_sorted_outputs_names[i];

        auto infer_stream_expected = m_infer_model->output(output_name);
        HAILO_CHECK_EXPECTED(infer_stream_expected, "Failed to get infer stream");

        infer_stream_expected->set_format_type(output_hailo_dtype);

        // We transform NCHW->NHWC / NHWC->NCHW in 'HailoKernel::infer()', and transform NHWC->device-format-order in libhailort
        // TODO: remove the double transformation when implementing transformations from/to NCHW
        if (hailo_format_order_t(format_order_params[i]) == HAILO_FORMAT_ORDER_NCHW) {
            infer_stream_expected->set_format_order(HAILO_FORMAT_ORDER_NHWC);
            m_output_should_double_order_conversion.push_back(true);
        } else {
            m_output_should_double_order_conversion.push_back(false);
        }
    }

    return HAILO_SUCCESS;
}

hailo_status HailoKernel::update_input_params(ConstPointerContainer<std::vector<NodeArg*>> &input_nodes,
    const std::vector<int64_t> &format_order_params)
{
    HAILO_ORT_ENFORCE(input_nodes.size() == m_sorted_inputs_names.size(),
        "Number of input nodes = ", input_nodes.size(), ", is inconsistent with input vstreams number = ", m_sorted_inputs_names.size());

    for (size_t i = 0; i < m_sorted_inputs_names.size(); i++) {
        auto input_ort_dtype = input_nodes[i]->TypeAsProto()->tensor_type().elem_type();
        auto input_hailo_dtype = HailoUtils::convert_ort_to_hailo_dtype(input_ort_dtype);
        auto input_name = m_sorted_inputs_names[i];

        auto infer_stream_expected = m_infer_model->input(input_name);
        HAILO_CHECK_EXPECTED(infer_stream_expected, "Failed to get infer stream");
        infer_stream_expected->set_format_type(input_hailo_dtype);

        // We transform NCHW->NHWC / NHWC->NCHW in 'HailoKernel::infer()', and transform NHWC->device-format-order in libhailort
        // TODO: remove the double transformation when implementing transformations from/to NCHW
        if (hailo_format_order_t(format_order_params[i]) == HAILO_FORMAT_ORDER_NCHW) {
            infer_stream_expected->set_format_order(HAILO_FORMAT_ORDER_NHWC);
            m_input_should_double_order_conversion.push_back(true);
        } else {
            m_input_should_double_order_conversion.push_back(false);
        }
    }

    return HAILO_SUCCESS;
}

std::chrono::milliseconds HailoKernel::get_infer_timeout_from_env_var()
{
    const char *env_value = std::getenv(HAILO_ONNXRT_INFER_TIMEOUT_MS_ENV_VAR);
    if (!env_value) {
        return INFER_DEFAULT_TIMEOUT;
    }

    std::istringstream iss(env_value);
    uint32_t timeout;

    HAILO_ORT_ENFORCE(iss >> timeout, "Failed parsing env var to uint32. '",
        HAILO_ONNXRT_INFER_TIMEOUT_MS_ENV_VAR, "' value: ", env_value);

    return std::chrono::milliseconds(timeout);
}

hailo_status HailoKernel::infer(OpKernelContext* context) const
{
    // TODO: HRT-6671 remove this after supportting multiple inputs.
    HAILO_ORT_ENFORCE(1 == m_infer_model->get_input_names().size(), "Multiple inputs is not supported.");
    hailo_status status;

    // TODO: HRT-6671 - When supporting multiple input we need to check that all frames_count in the shapes are the same.
    // Meanwhile, assuming that the number of frames is the same in the single input and the outputs
    size_t frames_count = context->Input<Tensor>(0)->Shape()[0];

    std::map<std::string, Tensor*> output_tensors;
    for (size_t i = 0; i < m_sorted_outputs_names.size(); i++) {
        auto output_name = m_sorted_outputs_names[i];
        auto infer_stream_expected = m_infer_model->output(output_name);
        HAILO_CHECK_EXPECTED(infer_stream_expected, "Failed to get infer stream");

        auto shape = infer_stream_expected->shape();
        auto format_order = infer_stream_expected->format().order;

        output_tensors.emplace(output_name, context->Output(i, HailoUtils::convert_hailo_shape(frames_count, shape, format_order)));
    }

    // Performing the inference frame by frame and without batch since,
    // currently sync_infer does not support multiple bindings (unlike async)
    for (size_t frame_id = 0 ; frame_id < frames_count ; frame_id++)
    {
        std::map<std::string, std::vector<uint8_t>> input_buffers;
        std::map<std::string, std::vector<uint8_t>> output_buffers;

        auto bindings_expected = m_configured_infer_model->create_bindings();
        HAILO_CHECK_EXPECTED(bindings_expected, "Failed to create bindings");
        auto bindings = bindings_expected.release();

        for (size_t i = 0; i < m_infer_model->get_input_names().size(); i++) {
            const auto *input_tensor = context->Input<Tensor>(i);
            HAILO_ORT_ENFORCE(nullptr != input_tensor, "input ", i, " is missing");

            auto input_name = m_infer_model->get_input_names()[0];
            auto infer_stream_expected = m_infer_model->input(input_name);
            HAILO_CHECK_EXPECTED(infer_stream_expected, "Failed to get infer stream");

            size_t input_frame_size = infer_stream_expected->get_frame_size();
            const uint8_t *current_frame_buffer_ptr = static_cast<const uint8_t*>(input_tensor->DataRaw()) + (frame_id * input_frame_size);

            if (m_input_should_double_order_conversion[i]) {
                hailo_3d_image_shape_t shape = infer_stream_expected->shape();

                input_buffers.emplace(input_name, std::vector<uint8_t>(input_frame_size));
                HailoUtils::transform_NCHW_to_NHWC(current_frame_buffer_ptr, input_buffers[input_name].data(), &shape,
                    infer_stream_expected->format().type, 1);

                auto input_data = MemoryView::create_const(input_buffers[input_name].data(), input_buffers[input_name].size());
                status = bindings.input(input_name)->set_buffer(input_data);
                HAILO_ORT_ENFORCE(HAILO_SUCCESS == status, "Failed to set infer input buffer");
            } else {
                auto input_data = MemoryView::create_const(current_frame_buffer_ptr, input_frame_size);
                status = bindings.input(input_name)->set_buffer(input_data);
                HAILO_ORT_ENFORCE(HAILO_SUCCESS == status, "Failed to set infer input buffer");
            }
        }

        for (size_t i = 0; i < m_sorted_outputs_names.size(); i++) {
            auto output_name = m_sorted_outputs_names[i];
            auto infer_stream_expected = m_infer_model->output(output_name);
            HAILO_CHECK_EXPECTED(infer_stream_expected, "Failed to get infer stream");

            size_t output_frame_size = infer_stream_expected->get_frame_size();

            if (m_output_should_double_order_conversion[i]) {
                output_buffers.emplace(output_name, std::vector<uint8_t>(output_frame_size));
                status = bindings.output(output_name)->set_buffer(MemoryView(output_buffers[output_name].data(), output_buffers[output_name].size()));
                HAILO_ORT_ENFORCE(HAILO_SUCCESS == status, "Failed to set infer output buffer");
            } else {
                uint8_t *current_tensor_ptr = static_cast<uint8_t*>(output_tensors[output_name]->MutableDataRaw()) + (frame_id * output_frame_size);
                status = bindings.output(output_name)->set_buffer(MemoryView(current_tensor_ptr, output_frame_size));
                HAILO_ORT_ENFORCE(HAILO_SUCCESS == status, "Failed to set infer output buffer");
            }
        }

        status = m_configured_infer_model->run(bindings, m_infer_timeout);
        HAILO_ORT_ENFORCE(HAILO_SUCCESS == status, "Failed to run infer");

        for (size_t i = 0; i < m_sorted_outputs_names.size(); i++) {
            if (!m_output_should_double_order_conversion[i]) {
                continue;
            }
            auto output_name = m_sorted_outputs_names[i];

            auto infer_stream_expected = m_infer_model->output(output_name);
            HAILO_CHECK_EXPECTED(infer_stream_expected, "Failed to get infer stream");

            auto output_frame_size = infer_stream_expected->get_frame_size();
            auto output_shape = infer_stream_expected->shape();
            auto output_format = infer_stream_expected->format();

            auto output_buffer_expected = bindings.output(output_name)->get_buffer();
            HAILO_CHECK_EXPECTED(output_buffer_expected, "Failed to get output buffer");

            uint8_t *current_tensor_ptr = static_cast<uint8_t*>(output_tensors[output_name]->MutableDataRaw()) + (frame_id * output_frame_size);

            HailoUtils::transform_NHWC_to_NCHW((void*)output_buffer_expected->data(),
                current_tensor_ptr, &output_shape, output_format.type, 1);
        }
    }
    return status;
}

Status HailoKernel::Compute(OpKernelContext* context) const
{
    auto status = infer(context);
    if (HAILO_SUCCESS == status) {
        return Status::OK();
    }
    else {
        return Status(common::ONNXRUNTIME, common::FAIL, "Error happend during inference, hailo status = "
            + std::to_string(status));
    }
}

}  // namespace onnxruntime