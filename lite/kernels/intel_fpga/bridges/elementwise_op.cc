// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <intelfpga.h>
#include "lite/kernels/intel_fpga/bridges/graph.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace intel_fpga {

int ElementwiseConverter(void *ctx, OpLite *op, KernelBase *kernel) {
    LOG(INFO) << "Converting elementwise op for intelfpga.";
    CHECK(ctx != nullptr);
    CHECK(op != nullptr);
    auto graph = static_cast<Graph*>(ctx);

    std::string op_type = op->op_info()->Type();
    auto op_info = op->op_info();
    auto scope = op->scope();

    auto input_x_name = op_info->Input("X").front();
    auto input_y_name = op_info->Input("Y").front();
    auto output_name = op_info->Output("Out").front();
    auto x_tensor = scope->FindMutableTensor(input_x_name);
    auto y_tensor = scope->FindMutableTensor(input_y_name);
    auto out_tensor = scope->FindMutableTensor(output_name);

    std::vector<float> x_scales;
    std::vector<float> y_scales;
    auto x_scale_name = "X0_scale";
    if (op_info->HasAttr("forced_scale")) {
      x_scales.push_back(op_info->GetAttr<float>(input_x_name + "_forced_scale"));
      y_scales.push_back(op_info->GetAttr<float>(input_y_name + "_forced_scale"));
    } else {
      if (op_info->HasInputScale(x_scale_name, true)) {
        x_scales = op_info->GetInputScale(x_scale_name, true);
      }
      auto y_scale_name = "Y0_scale";
      if (op_info->HasInputScale(y_scale_name, true)) {
        y_scales = op_info->GetInputScale(y_scale_name, true);
      }
    }
    LOG(INFO) << "X0_scale: " << x_scales[0];
    LOG(INFO) << "Y0_scale: " << y_scales[0];
    auto out_scale_name = "Out0_scale";
    std::vector<float> out_scales;
    //if (op_info->HasAttr("forced_scale")) {
    //  LOG(INFO) << "Fetch out scale from forced_scale.";
    //  auto scale =  op_info->GetAttr<float>("forced_scale");
    //  out_scales.push_back(scale);
    //} else if (op_info->HasOutputScale(out_scale_name, true)) {
    if (op_info->HasOutputScale(out_scale_name, true)) {
      out_scales = op_info->GetOutputScale(out_scale_name, true);
    }
    LOG(INFO) << "Out0_scale: " << out_scales[0];

    int32_t axis = op_info->GetAttr<int32_t>("axis");
    if (axis != -1) {
      LOG(FATAL) << "Only support axis =-1 for elementwise op.";
    }
    auto act_type =
      op_info->HasAttr("act_type") ? op_info->GetAttr<std::string>("act_type") : "";
    LOG(INFO) << "act_type: " << act_type;
    if (act_type != "" && act_type != "relu") {
      LOG(FATAL) << "Only support relu for activation for elementwise op.";
    }

    Node* node = new Node();
    node->is_output = graph->IsOutput(output_name);
    node->is_input = graph->IsInput(input_x_name);
    //    node->output_ref_count_ = 0;
    LOG(INFO) << "output_name: " << output_name;
    if (node->is_output) {
      LOG(INFO) << "output elementwise";
    }
    
    ElementwiseParam* ele_param = new ElementwiseParam();
    node->node_param_ = dynamic_cast<NodeParam*>(ele_param);
    if (act_type == "relu") {
      ele_param->ac_type = INTELFPGA_ACT_RELU;
    }
    ele_param->x_scale = x_scales[0];
    ele_param->y_scale = y_scales[0];
    ele_param->out_scale = out_scales[0];
    // Find this node's parent according to input tensor.
    if(graph->GetNodeByTensorName(input_x_name) &&
       graph->GetNodeByTensorName(input_y_name)) {
      node->parent_vec_.push_back(graph->GetNodeByTensorName(input_x_name));
      LOG(INFO) << "input_x_name: " << input_x_name;
      node->parent_vec_.push_back(graph->GetNodeByTensorName(input_y_name));
      LOG(INFO) << "input_y_name: " << input_y_name;
    } else if (graph->getGraphRootNode() == nullptr) {
      graph->setGraphRootNode(node); 
    }
    ele_param->input_x = x_tensor->mutable_data<int8_t>();
    ele_param->input_y = y_tensor->mutable_data<int8_t>();

    // Put this node's output tensor in map.
    graph->setTensor2Node(output_name, node);

    // Let predecessor node in topological order link to this node.
    auto pre_node = graph->getGraphTailNode();
    if(pre_node) {
    pre_node->next_ = node;
    }

    graph->setGraphTailNode(node);
    node->next_= nullptr;
    // Create node's device param.
    intelfpga_compute_s* device_param = new intelfpga_compute_s();
    node->device_param_ = device_param;

    //device_param->ia = x_tensor->data<int8_t>());
    device_param->oa = out_tensor->mutable_data<int8_t>();
    
    // Fill fpga_param.
    auto i_dims = x_tensor->dims();
    auto o_dims = out_tensor->dims();
    //init scale
    //device_param->scale = new float[2+2*o_dims[1]];
    //device_param->scale[0]= param.input_scale;
    //device_param->scale[1]= param.output_scale;
    //for(int i=0;i<o_dims[1];i++)
    //    device_param->scale[2+o_dims[1]+i]=0;

    device_param->param.input_offset = 0;
    device_param->param.scale_offset =  2;
    // device_param->param.weight_offset = 0;
    device_param->param.output_offset = 0;
    device_param->param.in_c=i_dims[1];
    device_param->param.in_h=i_dims[2];
    device_param->param.in_w=i_dims[3];
    device_param->param.output_c=o_dims[1];
    device_param->param.output_h=o_dims[2];
    device_param->param.output_w=o_dims[3];
    device_param->param.out_pad=0;
    // device_param->ip.dy = dilations[0];
    // device_param->ip.dx = dilations[1];
    device_param->param.type = INTELFPGA_ELE_ADD;
    LOG(INFO) << "Converting elementwise op end.";
    
  return SUCCESS;
}

}  // namespace intel_fpga
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    fusion_elementwise_add_activation,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    elementwise_add,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::ElementwiseConverter);