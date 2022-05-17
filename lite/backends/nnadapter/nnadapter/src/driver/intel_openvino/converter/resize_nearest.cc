// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/resize_nearest.h"
#include "driver/intel_openvino/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertResizeNearest(Converter* converter, core::Operation* operation) {
  RESIZE_NEAREST_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }

  std::shared_ptr<Tensor> shape_tensor(nullptr);
  std::shared_ptr<Tensor> scale_tensor(nullptr);
  if (shape_operand != nullptr) {
    shape_tensor = converter->GetMappedTensor(shape_operand);
    if (shape_tensor == nullptr) {
      shape_tensor = converter->ConvertOperand(shape_operand);
    }
  } else {
    scale_tensor = converter->GetMappedTensor(scales_operand);
    if (scale_tensor == nullptr) {
      scale_tensor = converter->ConvertOperand(scales_operand);
    }
  }

  ov::Interpolate::InterpolateAttrs attrs;
  attrs.mode = ov::Interpolate::InterpolateMode::NEAREST;
  attrs.nearest_mode = ov::Nearest_mode::SIMPLE;
  attrs.coordinate_transformation_mode = ov::CoordinateTransformMode::ASYMMETRIC;
  attrs.antialias = false;
  attrs.pads_begin = {0, 0, 0, 0};
  attrs.pads_end = {0, 0, 0, 0};

  // Get last 2 dimension of shape(input).
  auto shape_of_x = std::make_shared<ShapeOf>(input_tesnor, GetElementType<float>());
  auto shape_begin = converter->AddConstantTensor(std::vector<int64_t>({-2}));
  auto shape_end = converter->AddConstantTensor(std::vector<int64_t>({0}));
  auto input_hw_shape = std::make_shared<StridedSlice>(shape_of_x->output(0),
                                                            *shape_begin,
                                                            *shape_end,
                                                            std::vector<int64_t>{0},
                                                            std::vector<int64_t>{1});
  if(shape_tensor) {
    // Calculate scales from ouput shape.
    auto converted_shape = std::make_shared<default_opset::Convert>(*shape_tensor, GetElementType<float>());
    auto divide_op = std::make_shared<default_opset::Divide>(converted_shape->output(0), input_hw_shape->output(0));
    scale_tensor = std::make_shared<default_opset::Add>(divide_op->output(0), 1.0e-5);

  } else {
    // Calulate output shape from scales.
    auto multiply_op = std::make_shared<default_opset::Multiply>(input_hw_shape->output(0), *scale_tensor);
    shape_tensor = std::make_shared<default_opset::Convert>(multiply_op->output(0), GetElementType<int64_t>());
  }
  // Create axes tensor.
  int input_count = input_operand->type.dimensions.count;
  NNADAPTER_CHECK_GE(input_count, 2);
  auto axes_tensor = converter->AddConstantTensor<int64_t>({input_count - 2, input_count -1});
  auto interplate_op = std::make_shared<default_opset::Interpolate>(*input_tensor, *shape_tensor, *scale_tensor, *axes_tensor, attrs);
  MAP_OUTPUT(output_operand, interplate_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
