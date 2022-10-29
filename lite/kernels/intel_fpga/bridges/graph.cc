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

#include "lite/kernels/intel_fpga/bridges/graph.h"
#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {
namespace intel_fpga {



bool Graph::BuildDeviceModel() {
  auto cur_node = root_;
  while(cur_node) {
    if(cur_node->is_input){
      struct device_output_config config = intelfpga_output_malloc(
          &(cur_node->device_param_->d_ia),cur_node->device_param_->param.in_c,
          cur_node->device_param_->param.in_h,cur_node->device_param_->param.in_w);
        cur_node->device_param_->param.input_size=config.output_size;
        cur_node->device_param_->param.input_offset=config.output_offset;
    } else {
      cur_node->device_param_->param.input_size =
          cur_node->parent_vec_[0]->device_param_->param.output_size;
      cur_node->device_param_->param.input_offset =
          cur_node->parent_vec_[0]->device_param_->param.output_offset;
    }
    struct device_output_config config;
    // if(cur_node->device_param_->param.type==INTELFPGA_CALIB)
    //   config=intelfpga_calib_output_malloc(&(cur_node->device_param_->d_oa),cur_node->device_param_->param.output_c,
    //                                                     cur_node->device_param_->param.output_h,cur_node->device_param_->param.output_w);
    // else
		struct intelfpga_compute_s* argp = cur_node->device_param_;
    argp->param.output_row_tile=std::min(OUTPUT_BUFF_SIZE/argp->param.output_w,argp->param.output_h);
          argp->param.input_row_tile=(argp->param.output_row_tile-1)*argp->param.stride+argp->param.kernel;
          argp->param.output_channel_block_num =up_round(argp->param.output_c,OUTPUT_CHANNEL_TILE);
          argp->param.output_row_block_num=up_round(argp->param.output_h,argp->param.output_row_tile);
    config=intelfpga_output_malloc(&(cur_node->device_param_->d_oa),cur_node->device_param_->param.output_c,
                                                        cur_node->device_param_->param.output_h,cur_node->device_param_->param.output_w);
    cur_node->device_param_->param.output_size=config.output_size;
    cur_node->device_param_->param.output_offset=config.output_offset;

    cur_node=cur_node->next_;
  }
  FillAddParam();
  return true;
}

void Graph::FillAddParam() {
  LOG(INFO) << "Fill add op param.";
  auto cur_node = root_;
  while(cur_node) {
    if (cur_node->device_param_->param.type == INTELFPGA_ELE_ADD) {
      auto& add_param = dynamic_cast<ElementwiseParam*>(cur_node->node_param_)
                           ->add_param;
      add_param.input1_c = cur_node->parent_vec_[0]->device_param_
                                   ->param.in_c;
      add_param.input1_h = cur_node->parent_vec_[0]->device_param_
                                   ->param.in_h;
      add_param.input1_w = cur_node->parent_vec_[0]->device_param_
                                   ->param.in_w;
      add_param.input2_c = cur_node->parent_vec_[1]->device_param_
                                   ->param.in_c;
      add_param.input2_h = cur_node->parent_vec_[1]->device_param_
                                   ->param.in_h;
      add_param.input2_w = cur_node->parent_vec_[1]->device_param_
                                   ->param.in_w;
      add_param.output_c = cur_node->device_param_
                                   ->param.output_c;
      add_param.output_h = cur_node->device_param_
                                   ->param.output_h;
      add_param.output_w = cur_node->device_param_
                                   ->param.output_w;
      add_param.input1_scale = dynamic_cast<ElementwiseParam*>(cur_node->node_param_)
                                  ->x_scale;
      add_param.input2_scale = dynamic_cast<ElementwiseParam*>(cur_node->node_param_)
                                  ->y_scale;
      add_param.output_scale = dynamic_cast<ElementwiseParam*>(cur_node->node_param_)
                                  ->out_scale;
      add_param.input1_scale = add_param.input1_scale / add_param.output_scale;
      add_param.input2_scale = add_param.input2_scale / add_param.output_scale;
      add_param.type = int(BinaryOpType::op_add);
      add_param.relu = dynamic_cast<ElementwiseParam*>(cur_node->node_param_)
                                  ->ac_type;
      int input1_size = up_round(add_param.input1_c, INPUT_CHANNEL_TILE)
                           * INPUT_CHANNEL_TILE
                           * add_param.input1_h
                           * add_param.input1_w;
      int input2_size = up_round(add_param.input2_c, INPUT_CHANNEL_TILE)
                           * INPUT_CHANNEL_TILE
                           * add_param.input2_h
                           * add_param.input2_w;
      int output_size = up_round(add_param.output_c, INPUT_CHANNEL_TILE)
                           * INPUT_CHANNEL_TILE
                           * add_param.output_h
                           * add_param.output_w;
      add_param.input1_offset = (cur_node->parent_vec_[0]->device_param_
                                   ->param.output_offset)
                                   * INPUT_CHANNEL_TILE / ADD_INPUT_EXTEND_SCALE;
      add_param.input2_offset = (cur_node->parent_vec_[1]->device_param_
                                   ->param.output_offset)
                                   * INPUT_CHANNEL_TILE / ADD_INPUT_EXTEND_SCALE;
      add_param.output_offset = (cur_node->device_param_
                                    ->param.output_offset)
                                    * INPUT_CHANNEL_TILE / ADD_INPUT_EXTEND_SCALE;
    }
    cur_node=cur_node->next_;
  }
  LOG(INFO) << "Fill add op param done.";
}

bool Graph::DeviceModelValidCheck() {
  return true;
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
