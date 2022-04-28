// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "driver/intel_openvino/converter/converter.h"
#include "operation/arg_min_max.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertArgMinMax(Converter* converter, core::Operation* operation) {
  ARG_MIN_MAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto index_element_type = GetElementType<int64_t>();
  auto k = converter->AddConstantTensor<int64_t>({}, {1});
  auto axis_to_remove = converter->AddConstantTensor<uint64_t>(Shape{}, {static_cast<uint64_t>(axis)});
  auto node_topk = std::make_shared<default_opset::TopK>(*input_tensor, k, axis, "max", "index", index_element_type);
  auto reshaped_indices = std::make_shared<default_opset::Squeeze>(node_topk->output(1), axis_to_remove);
  auto convert_op = std::make_shared<default_opset::Convert>(reshaped_indices, index_element_type);
  auto output_tensor = MAP_OUTPUT(output_operand, convert_op, 0);
  if(keepdim) {
    auto axes_tensor = converter->AddConstantTensor<int32_t>({1}, {axis});
    auto unsqueeze_op = std::make_shared<default_opset::Unsqueeze>(convert_op, *axes_tensor);
    MAP_OUTPUT(output_operand, unsqueeze_op, 0);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
