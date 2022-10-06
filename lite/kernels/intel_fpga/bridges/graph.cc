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

  return true;
}

bool Graph::DeviceModelValidCheck() {
  return true;
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
