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

#include "lite/operators/conv_op.h"
#include<intelfpga.h>
#include "lite/kernels/intel_fpga/bridges/graph.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace intel_fpga {

int ConvConverter(void *ctx, OpLite *op, KernelBase *kernel) {
    CHECK(ctx != nullptr);
    CHECK(op != nullptr);
    auto graph = static_cast<Graph*>(ctx);

    operators::ConvParam& param = kernel->Param<operators::ConvParam>();
    std::string op_type = op->op_info()->Type();
    auto op_info = op->op_info();
    auto scope = op->scope();

    auto input_name = op_info->Input("Input").front();
    auto filter_name = op_info->Input("Filter").front();
    auto output_name = op_info->Output("Output").front();
    auto output = scope->FindMutableTensor(output_name);
    auto output_dims = output->dims();
    CHECK_EQ(output_dims[0], param.output->dims()[0]);
    CHECK_EQ(output_dims[1], param.output->dims()[1]);
    CHECK_EQ(output_dims[2], param.output->dims()[2]);
    CHECK_EQ(output_dims[3], param.output->dims()[3]);

    Node* node = new Node();
    node->is_output = graph->IsOutput(output_name);
    //    node->output_ref_count_ = 0;
    node->parent_ =nullptr;
    // Find this node's parent according to input tensor.
    if(graph->GetNodeByTensorName(input_name)) {
        node->parent_ = graph->GetNodeByTensorName(input_name);
        // Increat parent's output tensor reference.
        // (node->parent_)->output_ref_count_++;
    } else { //No parent. So mark this node as root.
      graph->setGraphRootNode(node);
    }

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

    device_param->ia = param.x->mutable_data<int8_t>();
    device_param->oa = param.output->mutable_data<int8_t>();
    device_param->ka = param.filter->mutable_data<int8_t>();
    float *ba = param.bias ? param.bias->mutable_data<float>() : nullptr;
    float *scale=param.weight_scale.data() ? param.weight_scale.data() : nullptr;

    // Fill fpga_param.
    auto w_dims = param.filter->dims();
    auto i_dims = param.x->dims();
    auto o_dims = param.output->dims();
    int group = param.groups;
    auto paddings = *param.paddings;
    auto dilations = *param.dilations;
    uint32_t at_;

    switch (param.activation_param.active_type) {
        case lite_api::ActivationType::kRelu:
        at_ = INTELFPGA_ACT_RELU;
        break;
        case lite_api::ActivationType::kRelu6:
        at_ = INTELFPGA_ACT_RELU6;
        break;
        case lite_api::ActivationType::kLeakyRelu:
        at_ = INTELFPGA_ACT_LEAKYRELU;
        device_param->lr = param.activation_param.Leaky_relu_alpha;
        break;
        default:
        at_ = INTELFPGA_ACT_NONE;
        break;
    }
    //init scale
    device_param->scale = new float[2+2*o_dims[1]];
    device_param->scale[0]= param.input_scale;
    device_param->scale[1]= param.output_scale;
    if(scale){
        for(int i=0;i<o_dims[1];i++)
        {
            device_param->scale[2+i]=scale[i];
        }
    }
    if(ba){
        for(int i=0;i<o_dims[1];i++)
        {
            device_param->scale[2+o_dims[1]+i]=ba[i]/param.output_scale;
        }
    }else{
        for(int i=0;i<o_dims[1];i++)
            device_param->scale[2+o_dims[1]+i]=0;
    }
    //ignore batch dimension TODO
    device_param->param.input_offset = 0;
    device_param->param.scale_offset =  2;
    device_param->d_ka = nullptr;
    device_param->param.output_offset = 0;
    device_param->param.in_c=i_dims[1];
    device_param->param.in_h=i_dims[2];
    device_param->param.in_w=i_dims[3];
    device_param->param.output_c=o_dims[1];
    device_param->param.output_h=o_dims[2];
    device_param->param.output_w=o_dims[3];
    device_param->param.in_pad=paddings[0];
    device_param->param.out_pad=0;
    device_param->param.kernel=w_dims[2];
    device_param->param.stride=param.strides[0];
    // device_param->param.output_row_tile=OUTPUT_BUFF_SIZE/o_dims[3]>o_dims[2]?o_dims[2]:OUTPUT_BUFF_SIZE/o_dims[3];
    // device_param->param.input_row_tile=(device_param->param.output_row_tile-1)*param.strides[0]+w_dims[2];
    // device_param->param.output_channel_block_num =(o_dims[1]-1)/OUTPUT_CHANNEL_TILE+1;
    // device_param->param.output_row_block_num=(o_dims[2]-1)/device_param->param.output_row_tile+1;
    // device_param->ip.dy = dilations[0];
    // device_param->ip.dx = dilations[1];
    device_param->param.relu=at_;
    device_param->param.type=(param.groups==1)?INTELFPGA_Conv2D:INTELFPGA_DW_Conv2D;
    fpga_init();
    if(param.groups==1){
        struct device_weight_config config= conv2d_weight_reorganize(
            device_param->ka,
            (int8_t**)(&(device_param->d_ka)),
            w_dims[0],
            w_dims[1],
            w_dims[2],
            w_dims[3],
            filter_name.c_str());
        device_param->param.weight_size = config.weight_size;
        device_param->param.weight_offset = config.weight_offset;
    }
    else{
        struct device_weight_config config = dw_conv2d_weight_reorganize(device_param->ka,(int8_t**)(&(device_param->d_ka)),w_dims[0],w_dims[2],w_dims[3]);
        device_param->param.weight_size = config.weight_size;
        device_param->param.weight_offset = config.weight_offset;
    }
  return SUCCESS;
}

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    conv2d,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::ConvConverter);

REGISTER_SUBGRAPH_BRIDGE(
    depthwise_conv2d,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::ConvConverter);
