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

#include "lite/operators/pool_op.h"
#include<intelfpga.h>
#include "lite/kernels/intel_fpga/bridges/graph.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace intel_fpga {

int PoolConverter(void *ctx, OpLite *op, KernelBase *kernel) {
    CHECK(ctx != nullptr);
    CHECK(op != nullptr);
    auto graph = static_cast<Graph*>(ctx);

    operators::PoolParam& param = kernel->Param<operators::PoolParam>();
    std::string op_type = op->op_info()->Type();
    auto op_info = op->op_info();
    auto scope = op->scope();

    auto input_name = op_info->Input("X").front();
    auto output_name = op_info->Output("Out").front();

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
    node->next_=NULL;
    // Create node's device param.
    intelfpga_compute_s* device_param = new intelfpga_compute_s();
    node->device_param_ = device_param;

    device_param->ia = (int8_t*)(param.x->data<int8_t>());
    device_param->oa = (int8_t*)(param.output->mutable_data<float>());
    
    // Fill fpga_param.
    auto i_dims = param.x->dims();
    auto o_dims = param.output->dims();
    auto paddings = *param.paddings; 
    //init scale
    device_param->scale = new float[2+2*o_dims[1]];
    device_param->scale[0]= param.input_scale;
    device_param->scale[1]= param.output_scale;
    for(int i=0;i<o_dims[1];i++)
        device_param->scale[2+o_dims[1]+i]=0;

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
    device_param->param.in_pad=paddings[0];
    device_param->param.out_pad=0;
    device_param->param.stride=param.strides[0];
    device_param->param.kernel=param.ksize[0];
    // device_param->ip.dy = dilations[0];
    // device_param->ip.dx = dilations[1];
    if(param.pooling_type == "max"){
        device_param->param.type=INTELFPGA_Pool2D_MAX;
    }
    else{
        device_param->param.type=INTELFPGA_Pool2D_AVG;
    }
    
  return SUCCESS;
}

}  // namespace intel_fpga
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    pool2d,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::PoolConverter);

