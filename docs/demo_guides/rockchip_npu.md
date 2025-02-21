# 瑞芯微 NPU

Paddle Lite 已支持 Rockchip 1代 NPU 的预测部署。
其接入原理是与之前华为 Kirin NPU 类似，即加载并分析 Paddle 模型，首先将 Paddle 算子转成 NNAdapter 标准算子，其次再转换为 Rockchip NPU 组网 API 进行网络构建，在线生成并执行模型。
- **请注意**：本文介绍的是 Paddle Lite 基于 RK DDK 来调用瑞芯微 SoC 的 NPU 算力，考虑到算子以及模型支持的广度，如果需要在瑞芯微 SoC 上部署较为复杂的模型，我们强烈建议您参考[芯原 TIM-VX 部署示例](./verisilicon_timvx)，同样能调用晶晨 SoC 的 NPU 算力，且支持场景更多。
## 支持现状

### 已支持的芯片

- RK1808/1806
- RV1126/1109
注意：暂时不支持 RK3399Pro

### 已支持的设备

- RK1808/1806 EVB
- TB-RK1808S0 AI 计算棒
- RV1126/1109 EVB

### 已支持的 Paddle 模型

#### 模型
- [mobilenet_v1_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz)
- [resnet50_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/resnet50_int8_224_per_layer.tar.gz)
- [ssd_mobilenet_v1_relu_voc_int8_300_per_layer](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_relu_voc_int8_300_per_layer.tar.gz)

#### 性能
- 测试环境
  - 编译环境
    - Ubuntu 16.04，GCC 5.4 for ARMLinux armhf and aarch64

  - 硬件环境
    - RK1808EVB/TB-RK1808S0 AI 计算棒
      - CPU：2 x Cortex-A35 1.6 GHz
      - NPU：3 TOPs for INT8 / 300 GOPs for INT16 / 100 GFLOPs for FP16

    - RV1109EVB
      - CPU：2 x Cortex-A7 1.2 GHz
      - NPU：1.2Tops，support INT8/ INT16

- 测试方法
  - warmup=1, repeats=5，统计平均时间，单位是 ms
  - 线程数为 1，`paddle::lite_api::PowerMode CPU_POWER_MODE` 设置为 ` paddle::lite_api::PowerMode::LITE_POWER_HIGH`
  - 分类模型的输入图像维度是{1, 3, 224, 224}，检测模型的维度是{1, 3, 300, 300}

- 测试结果

  |模型 |RK1808EVB||TB-RK1808S0 AI 计算棒||RV1109EVB||
  |---|---|---|---|---|---|---|
  |  |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |
  |mobilenet_v1_int8_224_per_layer|  266.965796|  6.982800|  357.467200|  9.330400|  331.796204|  7.494000|
  |resnet50_int8_224_per_layer|  1503.052393|  19.387600|  2016.901196|  22.655600|  1959.528223|  30.797000|
  |ssd_mobilenet_v1_relu_voc_int8_300_per_layer|  545.154004|  15.315800|  731.145203|  19.800800|  696.48919|  14.957600|

### 已支持（或部分支持）NNAdapter 的 Paddle 算子
您可以查阅[ NNAdapter 算子支持列表](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/converter/all.h)获得各算子在不同新硬件上的最新支持信息。

**不经过 NNAdapter 标准算子转换，而是直接将 Paddle 算子转换成 `Rockchip NPU IR` 的方案可点击[链接](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/rockchip_npu.html)**。

## 参考示例演示

### 测试设备

- RK1808 EVB

  ![rk1808_evb_front](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rk1808_evb_front.jpg)

  ![rk1808_evb_back](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rk1808_evb_back.jpg)

- TB-RK1808S0 AI计算棒

  ![tb-rk1808s0](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/TB-RK1808S0.jpg)

- RV1126 EVB

   ![rk1126_evb](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/rv1126_evb.jpg)

### 准备设备环境

- RK1808 EVB

  - 需要依赖特定版本的 firmware，请参照[ rknpu_ddk ](https://github.com/airockchip/rknpu_ddk)的说明对设备进行 firmware 的更新；
  - 由于 RK1808 EVB 在刷 firmware 后，只是一个纯净的 Linux 系统，无法像 Ubuntu 那样使用 `apt-get` 命令方便的安装软件，因此，示例程序和 Paddle Lite 库的编译均采用交叉编译方式；
  - 将 `MicroUSB` 线插入到设备的 `MicroUSB OTG` 口，就可以使用 Android 的 `adb` 命令进行设备的交互，再也不用配置网络使用 `ssh` 或者通过串口的方式访问设备了，这个设计非常赞！
  - **将 rknpu_ddk 的 `lib64` 目录下除 `librknpu_ddk.so` 之外的动态库都拷贝到设备的 `/usr/lib` 目录下，更新 Rockchip NPU 的系统库。**
  - **注意确认 Galcore 驱动版本，需为 6.4.0.X 方能正常运行。 Galcore 由开发板/解决方案厂商提供，在刷新固件时也会同时刷新 Galcore 驱动**
    ```shell
    $ dmesg | grep Galcore
    [   15.978465] Galcore version 6.4.0.227915
    ```

- TB-RK1808S0 AI 计算棒

  - 参考[ TB-RK1808S0 wiki 教程的](https://t.rock-chips.com/wiki.php?filename=%E6%9D%BF%E7%BA%A7%E6%8C%87%E5%8D%97/TB-RK1808S0)将计算棒配置为主动模式，完成网络设置和 firmware 的升级，具体步骤如下：
    - 将计算棒插入 Window7/10 主机，参考[主动模式开发](https://t.rock-chips.com/wiki.php?filename=%E6%9D%BF%E7%BA%A7%E6%8C%87%E5%8D%97/TB-RK1808S0#hash_6)配主机的虚拟网卡 IP 地址，通过 `ssh toybrick@192.168.180.8` 验证是否能登录计算棒；
    - 参考[ Window7/10 系统配置计算棒网络共享](https://t.rock-chips.com/wiki.php?filename=%E6%9D%BF%E7%BA%A7%E6%8C%87%E5%8D%97/TB-RK1808S0#hash_7)，`SSH` 登录计算棒后通过 `wget www.baidu.com` 验证是否能够访问外网；
    - 参考[固件在线升级](https://t.rock-chips.com/wiki.php?filename=%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8/%E7%83%A7%E5%86%99%E5%9B%BA%E4%BB%B6)，建议通过 `ssh` 登录计算棒，在 `shell` 下执行 `sudo dnf update -y` 命令快速升级到最新版本系统（要求系统版本 >= 1.4.1-2），可通过 `rpm -qa | grep toybrick-server` 查询系统版本：

    ```shell
    $ rpm -qa | grep toybrick-server
    toybrick-server-1.4.1-2.rk1808.fc28.aarch64
    ```
    - **将 rknpu_ddk 的 `lib64` 目录下除 `librknpu_ddk.so` 之外的动态库都拷贝到设备的 `/usr/lib` 目录下，更新 Rockchip NPU 的系统库。**
    - **注意确认 Galcore 驱动版本，需为 6.4.0.X 方能正常运行。 Galcore 由开发板/解决方案厂商提供，在刷新固件时也会同时刷新 Galcore 驱动**
    ```shell
    $ dmesg | grep Galcore
    [    7.919345] Galcore version 6.4.0.227915
    ```
- RV1126 EVB

   - 需要升级 1.51 的 firmware（下载和烧录方法请联系RK相关同学），可通过以下命令确认 librknn_runtime.so 的版本：

    ```shell
    # strings /usr/lib/librknn_runtime.so | grep build |grep version
    librknn_runtime version 1.5.1 (161f53f build: 2020-11-05 15:12:30 base: 1126)
    ```

   - 示例程序和 Paddle Lite 库的编译需要采用交叉编译方式，通过 `adb` 进行设备的交互和示例程序的运行。
   - **将 rknpu_ddk 的 `lib64` 目录下除 `librknpu_ddk.so` 之外的动态库都拷贝到设备的 `/usr/lib` 目录下，更新 Rockchip NPU 的系统库。**
  - **注意确认 Galcore 驱动版本，需为 6.4.0.X 方能正常运行。 Galcore 由开发板/解决方案厂商提供，在刷新固件时也会同时刷新 Galcore 驱动**
    ```shell
    $ dmesg | grep Galcore
    [    5.809874] Galcore version 6.4.0.227915
    ```

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[编译环境准备](../source_compile/compile_env)中的 Docker 开发环境进行配置；
- 由于有些设备只提供网络访问方式（例如：TB-RK1808S0 AI 计算棒），需要通过 `scp` 和 `ssh` 命令将交叉编译生成的 Paddle Lite 库和示例程序传输到设备上执行，因此，在进入 Docker 容器后还需要安装如下软件：

  ```
  # apt-get install openssh-client sshpass
  ```

### 运行图像分类示例程序

- 下载 Paddle Lite 通用示例程序[ PaddleLite-generic-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下：

  ```shell
    - PaddleLite-generic-demo
      - image_classification_demo
        - assets
          - images
            - tabby_cat.jpg # 测试图片
            - tabby_cat.raw # 经过 convert_to_raw_image.py 处理后的 RGB Raw 图像
          - labels
            - synset_words.txt # 1000 分类 label 文件
          - models
            - mobilenet_v1_int8_224_per_layer
              - __model__ # Paddle fluid 模型组网文件，可使用 netron 查看网络结构
              — conv1_weights # Paddle fluid 模型参数文件
              - batch_norm_0.tmp_2.quant_dequant.scale # Paddle fluid 模型量化参数文件
              — subgraph_partition_config_file.txt # 自定义子图分割配置文件
              ...
        - shell
          - CMakeLists.txt # 示例程序 CMake 脚本
          - build.linux.arm64 # arm64 编译工作目录
            - image_classification_demo # 已编译好的，适用于 arm64 的示例程序
          - build.linux.armhf # armhf编译工作目录
            - image_classification_demo # 已编译好的，适用于 armhf 的示例程序
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序ssh运行脚本
          - run_with_adb.sh # 示例程序adb运行脚本
      - libs
        - PaddleLite
          - linux
            - arm64 # Linux 64 位系统
              - include # Paddle Lite 头文件
              - lib # Paddle Lite 库文件
                - rockchip_npu # 瑞芯微 NPU DDK、NNAdapter 运行时库、device HAL 库
                	- libnnadapter.so # NNAdapter 运行时库
                	- librockchip_npu.so # NNAdapter device HAL 库
                  - librknpu_ddk.so # 瑞芯微 NPU DDK
                  - libGAL.so # 芯原 DDK
                  - libVSC.so # 芯原 DDK
                  - libOpenVX.so # 芯原 DDK
                  - libgomp.so.1 # gnuomp 库
                - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            - armhf # Linux 32 位系统
              - include
              - lib
                - rockchip_npu
                  - libnnadapter.so
                  - librockchip_npu.so
                  - librknpu_ddk.so
                  - libGAL.so
                  - libVSC.so
                  - libOpenVX.so
                  ...
            	  - libpaddle_full_api_shared.so # 预编译 Paddle Lite full api 库
                - libpaddle_light_api_shared.so # 预编译 Paddle Lite light api 库
            ...
          - android
        - OpenCV # OpenCV 预编译库
      - ssd_detection_demo # 基于 ssd 的目标检测示例程序
  ```

- 按照以下命令分别运行转换后的 ARM CPU 模型和 Rockchip NPU 模型，比较它们的性能和结果；

  ```shell
  注意：
  1）`run_with_adb.sh` 不能在 Docker 环境执行，否则可能无法找到设备，也不能在设备上运行。
  2）`run_with_ssh.sh` 不能在设备上运行，且执行前需要配置目标设备的IP地址、SSH账号和密码。
  3）`build.sh` 根据入参生成针对不同操作系统、体系结构的二进制程序，需查阅注释信息配置正确的参数值。
  4）`run_with_adb.sh` 入参包括模型名称、操作系统、体系结构、目标设备、设备序列号等，需查阅注释信息配置正确的参数值。
  5）`run_with_ssh.sh` 入参包括模型名称、操作系统、体系结构、目标设备、ip 地址、用户名、用户密码等，需查阅注释信息配置正确的参数值。
  6）下述命令行示例中涉及的具体IP、SSH账号密码、设备序列号等均为示例环境，请用户根据自身实际设备环境修改。

  在 ARM CPU 上运行 mobilenet_v1_int8_224_per_layer 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell

  For RK1808 EVB
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu a133d8abb26137b2
    (RK1808 EVB)
    warmup: 1 repeat: 5, average: 266.965796 ms, max: 267.056000 ms, min: 266.848999 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 2.423000 ms
    Prediction time: 266.965796 ms
    Postprocess time: 0.538000 ms

  For RK1806/RV1126/RV1109 EVB
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux armhf cpu 192.168.100.13 22 root rockchip
    (RV1109 EVB)
    warmup: 1 repeat: 5, average: 331.796204 ms, max: 341.756012 ms, min: 328.386993 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 3.380000 ms
    Prediction time: 331.796204 ms
    Postprocess time: 0.554000 ms

  For TB-RK1808S0 AI 计算棒
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu 192.168.180.8 22 toybrick toybrick
    (TB-RK1808S0 AI计算棒)
    warmup: 1 repeat: 5, average: 357.467200 ms, max: 358.815002 ms, min: 356.808014 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 3.199000 ms
    Prediction time: 357.467200 ms
    Postprocess time: 0.596000 ms

  ------------------------------

  在 Rockchip NPU 上运行 mobilenet_v1_int8_224_per_layer 全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell

  For RK1808 EVB
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 rockchip_npu a133d8abb26137b2
    (RK1808 EVB)
    warmup: 1 repeat: 5, average: 6.982800 ms, max: 7.045000 ms, min: 6.951000 ms
    results: 3
    Top0  Egyptian cat - 0.514779
    Top1  tabby, tabby cat - 0.421183
    Top2  tiger cat - 0.052648
    Preprocess time: 2.417000 ms
    Prediction time: 6.982800 ms
    Postprocess time: 0.509000 ms

  For RK1806/RV1126/RV1109 EVB
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux armhf rockchip_npu 192.168.100.13 22 root rockchip
    (RV1109 EVB)
    warmup: 1 repeat: 5, average: 7.494000 ms, max: 7.724000 ms, min: 7.321000 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 3.532000 ms
    Prediction time: 7.494000 ms
    Postprocess time: 0.577000 ms

  For TB-RK1808S0 AI 计算棒
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 rockchip_npu 192.168.180.8 22 toybrick toybrick
    (TB-RK1808S0 AI 计算棒)
    warmup: 1 repeat: 5, average: 9.330400 ms, max: 9.753000 ms, min: 8.421000 ms
    results: 3
    Top0  Egyptian cat - 0.514779
    Top1  tabby, tabby cat - 0.421183
    Top2  tiger cat - 0.052648
    Preprocess time: 3.170000 ms
    Prediction time: 9.330400 ms
    Postprocess time: 0.634000 ms
  ```

- 如果需要更改测试图片，可将图片拷贝到 `PaddleLite-generic-demo/image_classification_demo/assets/images` 目录下，然后调用 `convert_to_raw_image.py` 生成相应的 RGB Raw 图像，最后修改 `run_with_adb.sh`、`run_with_ssh.sh` 的 IMAGE_NAME 变量即可；
- 重新编译示例程序：  
  ```shell
  注意：
  1）请根据 `buid.sh`配置正确的参数值。
  2）需在 Docker 环境中编译。

  # 对于 RK1808EVB, TB-RK1808S0
  ./build.sh linux arm64

  # 对于 RK1806EVB, RV1109/1126 EVB
  ./build.sh linux armhf
  ```

### 更新模型
- 通过 Paddle 训练或 X2Paddle 转换得到 MobileNetv1 foat32 模型[ mobilenet_v1_fp32_224 ](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)
- 通过 Paddle+PaddleSlim 后量化方式，生成[ mobilenet_v1_int8_224_per_layer量化模型 ](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/mobilenet_v1_int8_224_fluid.tar.gz)
- 下载[ PaddleSlim-quant-demo.tar.gz ](https://paddlelite-demo.bj.bcebos.com/tools/PaddleSlim-quant-demo.tar.gz)，解压后清单如下：
    ```shell
    - PaddleSlim-quant-demo
      - image_classification_demo
        - quant_post # 后量化
          - quant_post_rockchip_npu.sh # Rockchip NPU 一键量化脚本
          - README.md # 环境配置说明，涉及 PaddlePaddle、PaddleSlim 的版本选择、编译和安装步骤
          - datasets # 量化所需要的校准数据集合
            - ILSVRC2012_val_100 # 从 ImageNet2012 验证集挑选的 100 张图片
          - inputs # 待量化的 fp32 模型
            - mobilenet_v1
            - resnet50
          - outputs # 产出的全量化模型
          - scripts # 后量化内置脚本
    ```
- 查看 `README.md` 完成 PaddlePaddle 和 PaddleSlim 的安装
- 直接执行 `./quant_post_rockchip_npu.sh` 即可在 `outputs` 目录下生成 mobilenet_v1_int8_224_per_layer 量化模型
  ```shell
  -----------  Configuration Arguments -----------
  activation_bits: 8
  activation_quantize_type: moving_average_abs_max
  algo: KL
  batch_nums: 10
  batch_size: 10
  data_dir: ../dataset/ILSVRC2012_val_100
  is_full_quantize: 1
  is_use_cache_file: 0
  model_path: ../models/mobilenet_v1
  optimize_model: 1
  output_path: ../outputs/mobilenet_v1
  quantizable_op_type: conv2d,depthwise_conv2d,mul
  use_gpu: 0
  use_slim: 1
  weight_bits: 8
  weight_quantize_type: abs_max
  ------------------------------------------------
  quantizable_op_type:['conv2d', 'depthwise_conv2d', 'mul']
  2021-08-30 05:52:10,048-INFO: Load model and set data loader ...
  2021-08-30 05:52:10,129-INFO: Optimize FP32 model ...
  I0830 05:52:10.139564 14447 graph_pattern_detector.cc:91] ---  detected 14 subgraphs
  I0830 05:52:10.148236 14447 graph_pattern_detector.cc:91] ---  detected 13 subgraphs
  2021-08-30 05:52:10,167-INFO: Collect quantized variable names ...
  2021-08-30 05:52:10,168-WARNING: feed is not supported for quantization.
  2021-08-30 05:52:10,169-WARNING: fetch is not supported for quantization.
  2021-08-30 05:52:10,170-INFO: Preparation stage ...
  2021-08-30 05:52:11,853-INFO: Run batch: 0
  2021-08-30 05:52:16,963-INFO: Run batch: 5
  2021-08-30 05:52:21,037-INFO: Finish preparation stage, all batch:10
  2021-08-30 05:52:21,048-INFO: Sampling stage ...
  2021-08-30 05:52:31,800-INFO: Run batch: 0
  2021-08-30 05:53:23,443-INFO: Run batch: 5
  2021-08-30 05:54:03,773-INFO: Finish sampling stage, all batch: 10
  2021-08-30 05:54:03,774-INFO: Calculate KL threshold ...
  2021-08-30 05:54:28,580-INFO: Update the program ...
  2021-08-30 05:54:29,194-INFO: The quantized model is saved in ../outputs/mobilenet_v1
  post training quantization finish, and it takes 139.42292165756226.

  -----------  Configuration Arguments -----------
  batch_size: 20
  class_dim: 1000
  data_dir: ../dataset/ILSVRC2012_val_100
  image_shape: 3,224,224
  inference_model: ../outputs/mobilenet_v1
  input_img_save_path: ./img_txt
  save_input_img: False
  test_samples: -1
  use_gpu: 0
  ------------------------------------------------
  Testbatch 0, acc1 0.8, acc5 1.0, time 1.63 sec
  End test: test_acc1 0.76, test_acc5 0.92
  --------finish eval int8 model: mobilenet_v1-------------
  ```
  - 参考[模型转化方法](../user_guides/model_optimize_tool)，利用 opt 工具转换生成 Rockchip NPU 模型，仅需要将 `valid_targets` 设置为 rockchip_npu,arm 即可。
  ```shell
  $ ./opt --model_dir=mobilenet_v1_int8_224_per_layer \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=rockchip_npu,arm
  ```
### 更新支持 Rockchip NPU 的 Paddle Lite 库

- 下载 Paddle Lite 源码和 Rockchip NPU DDK

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  $ git clone https://github.com/airockchip/rknpu_ddk.git
  ```

- 编译并生成 `PaddleLite+RockchipNPU` for armv8 and armv7 的部署库

  - For RK1808 EVB and TB-RK1808S0 AI计算棒
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_rockchip_npu=ON --nnadapter_rockchip_npu_sdk_root=$(pwd)/rknpu_ddk

      ```
    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_rockchip_npu=ON --nnadapter_rockchip_npu_sdk_root=$(pwd)/rknpu_ddk full_publish

      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/rockchip_npu/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/librockchip_npu.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/rockchip_npu/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

  - For RK1806/RV1126/RV1109 EVB
    - tiny_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_rockchip_npu=ON --nnadapter_rockchip_npu_sdk_root=$(pwd)/rknpu_ddk
      ```

    - full_publish 编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --arch=armv7hf --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_rockchip_npu=ON --nnadapter_rockchip_npu_sdk_root=$(pwd)/rknpu_ddk full_publish
      ```
    - 替换头文件和库
      ```shell
      # 替换 include 目录
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      # 替换 NNAdapter 运行时库
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/rockchip_npu/
      # 替换 NNAdapter device HAL 库
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/librockchip_npu.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/rockchip_npu/
      # 替换 libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      # 替换 libpaddle_full_api_shared.so (仅在 full_publish 编译方式下)
      $ cp -rf build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/lib/
      ```
  
- 替换头文件后需要重新编译示例程序

## 其它说明

- RK研发同学正在持续增加用于适配 Paddle 算子 `bridge/converter`，以便适配更多 Paddle 模型。
