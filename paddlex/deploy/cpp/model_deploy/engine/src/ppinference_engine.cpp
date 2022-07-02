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

#include "model_deploy/engine/include/ppinference_engine.h"
#include "model_deploy/common/include/logger.h"

namespace PaddleDeploy {
bool Model::PaddleEngineInit(const PaddleEngineConfig& engine_config) {
  infer_engine_ = std::make_shared<PaddleInferenceEngine>();
  InferenceConfig config("paddle");
  *(config.paddle_config) = engine_config;
  return infer_engine_->Init(config);
}

bool PaddleInferenceEngine::Init(const InferenceConfig& infer_config) {
  const PaddleEngineConfig& engine_config = *(infer_config.paddle_config);

  // 第一轮auto tune
  if (engine_config.use_trt && engine_config.use_gpu) {
      paddle_infer::Config _config;
      _config.SetModel(engine_config.model_filename, engine_config.params_filename);
      _config.EnableUseGpu(100, 0);
      _config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kFloat32, false, false);
      _config.CollectShapeRangeInfo(engine_config.shape_range_info_path);
      _config.DisableGlogInfo();
      auto predictor = paddle_infer::CreatePredictor(_config);
      // 准备数据
      cv::Mat img = cv::Mat::ones(cv::Size(engine_config.target_width, engine_config.target_height), CV_8UC3);

      //LOGC("Info", "shape range info path: %s", engine_config.shape_range_info_path);
      //LOGC("Info", "img size: %d, %d", img.rows, img.cols);
      //LOGC("Info", "target size: %d, %d", engine_config.target_width, engine_config.target_height);
      int rows, cols, chs;
      std::vector<float> img_data;
      // 对图像进行预处理：resize和normalize，
      img.convertTo(img, CV_32F, 1.0 / 255, 0);
      img = (img - 0.5) / 0.5;
      rows = img.rows;
      cols = img.cols;
      chs = img.channels();
      img_data.resize(rows * cols * chs);
      // hwc to chw
      for (int i = 0; i < chs; ++i) {
          cv::extractChannel(img, cv::Mat(rows, cols, CV_32FC1, img_data.data() + i * rows * cols), i);
      }
      // 准备input
      auto input_names = predictor->GetInputNames();
      if(engine_config.model_type == "seg"){
        auto input_t = predictor->GetInputHandle(input_names[0]);
        std::vector<int> input_shape = { 1, chs, rows, cols };
        input_t->Reshape(input_shape);
        input_t->CopyFromCpu(img_data.data());
      }
      else if (engine_config.model_type == "det") {
          for (auto name : input_names) {
              /* DetInput
               * vector<float>img_data=resize(rows*cols*chs), 
                 vector<float>img_shape={float rows, cols}, 
                 vector<float>scale_factor={float 1, 1}, 
                 vector<float>in_net_shape={float rows, cols}
               */
              if (name == "im_shape") {
                  auto input_shape = predictor->GetInputHandle(name);
                  std::vector<float> img_shape = { static_cast<float>(rows), static_cast<float>(cols) };
                  input_shape->Reshape(std::vector<int>{1, 2});
                  input_shape->CopyFromCpu(img_shape.data());
              }
              else if (name == "image") {
                  auto input_img = predictor->GetInputHandle(name);
                  std::vector<float> in_net_shape = { static_cast<float>(rows), static_cast<float>(cols) };
                  input_img->Reshape(std::vector<int>{1, 3, static_cast<int>(in_net_shape[0]), static_cast<int>(in_net_shape[1])});
                  input_img->CopyFromCpu(img_data.data());
              }
              else if (name == "scale_factor") {
                  auto input_scale = predictor->GetInputHandle(name);
                  std::vector<float> scale_factor = { static_cast<float>(1.0), static_cast<float>(1.0) };
                  input_scale->Reshape(std::vector<int>{1, 2});
                  input_scale->CopyFromCpu(scale_factor.data());
              }
          }
      }
      // 执行一次前向计算得到shape info
      predictor->Run();
      LOGC("Info", "saved shape range info to pbtxt file.");
  }

  // 第二轮正式predictor
  paddle_infer::Config config;
  if ("" == engine_config.key) {
    config.SetModel(engine_config.model_filename,
                  engine_config.params_filename);
  } else {
#ifdef PADDLEX_DEPLOY_ENCRYPTION
    std::string model = decrypt_file(engine_config.model_filename.c_str(),
                                     engine_config.key.c_str());
    std::string params = decrypt_file(engine_config.params_filename.c_str(),
                                      engine_config.key.c_str());
    config.SetModelBuffer(model.c_str(),
                          model.size(),
                          params.c_str(),
                          params.size());
#else
    std::cerr << "Don't open with_encryption on compile" << std::endl;
    return false;
#endif  // PADDLEX_DEPLOY_ENCRYPTION
  }
  LOGC("Info", "set model file position: %s,%s", engine_config.model_filename.c_str(), engine_config.params_filename.c_str());
  if (engine_config.use_mkl && !engine_config.use_gpu) {
    config.EnableMKLDNN();
    config.SetCpuMathLibraryNumThreads(engine_config.mkl_thread_num);
    config.SetMkldnnCacheCapacity(10);
  }
  LOGC("Info", "set use mkl: %d, use_gpu:%d", (int)engine_config.use_mkl, (int)engine_config.use_gpu);
  if (engine_config.use_gpu) {
    config.EnableUseGpu(100, engine_config.gpu_id);
  } else {
    config.DisableGpu();
  }
  LOGC("Info", "set use gpu: %d", (int)engine_config.use_gpu);

  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  config.SwitchIrOptim(engine_config.use_ir_optim);
  config.EnableMemoryOptim();
  LOGC("Info", "set mem optim");

  if (engine_config.use_trt && engine_config.use_gpu) {
    paddle_infer::PrecisionType precision;
    if (engine_config.precision == 0) {
      precision = paddle_infer::PrecisionType::kFloat32;
    } else if (engine_config.precision == 1) {
      precision = paddle_infer::PrecisionType::kHalf;
    } else if (engine_config.precision == 2) {
      precision = paddle_infer::PrecisionType::kInt8;
    } else {
      std::cerr << "Can not support the set precision" << std::endl;
      return false;
    }
    LOGC("Info", "set precision");

    config.EnableTensorRtEngine(
        engine_config.max_workspace_size /* workspace_size*/,
        engine_config.max_batch_size /* max_batch_size*/,
        engine_config.min_subgraph_size /* min_subgraph_size*/,
        precision /* precision*/,
        engine_config.use_static /* use_static*/,
        engine_config.use_calib_mode /* use_calib_mode*/);
    LOGC("Info", "set enable trt engine:: maxworkspacesize:%d, maxbatchsize:%d, minsubsize:%d, use_static:%d, usecalibmod:%d", 
        engine_config.max_workspace_size, engine_config.max_batch_size, engine_config.min_subgraph_size, (int)engine_config.use_static, (int)engine_config.use_calib_mode);

    // [suliang] 增加判断是否采用auto tune
    if (engine_config.min_input_shape.size() != 0 && engine_config.shape_range_info_path == "") {
      config.SetTRTDynamicShapeInfo(engine_config.min_input_shape,
                                    engine_config.max_input_shape,
                                    engine_config.optim_input_shape);
      LOGC("Info", "handly set dynamic shape");
    }
    else {
      //[suliang] 增加读取shape info文件
      LOGC("Info", "set shape range path: %s", engine_config.shape_range_info_path.c_str());
      config.EnableTunedTensorRtDynamicShape(engine_config.shape_range_info_path, true);
    }    
  }
  LOGC("Info", "start create predictor");
  predictor_ = std::move(paddle_infer::CreatePredictor(config));
  LOGC("Info", "finish create predictor");
  return true;
}

bool PaddleInferenceEngine::Infer(const std::vector<DataBlob>& inputs,
                                  std::vector<DataBlob>* outputs) {
  LOGC("Info", "input image size:%d", inputs.size());
  LOGC("Info", "input[0].shape: %d, %d", inputs.size(), inputs[0].shape[0], inputs[0].shape[1]);
  LOGC("Info", "input image dtype:%d", inputs[0].dtype);
  if (inputs.size() == 0) {
    std::cerr << "empty input image on PaddleInferenceEngine" << std::endl;
    return true;
  }
  auto input_names = predictor_->GetInputNames();
  for (int i = 0; i < inputs.size(); i++) {
    auto in_tensor = predictor_->GetInputHandle(input_names[i]);
    in_tensor->Reshape(inputs[i].shape);
    if (inputs[i].dtype == FLOAT32) {
      float* im_tensor_data;
      im_tensor_data = (float*)(inputs[i].data.data());  // NOLINT
      in_tensor->CopyFromCpu(im_tensor_data);
    } else if (inputs[i].dtype == INT64) {
      int64_t* im_tensor_data;
      im_tensor_data = (int64_t*)(inputs[i].data.data());  // NOLINT
      in_tensor->CopyFromCpu(im_tensor_data);
    } else if (inputs[i].dtype == INT32) {
      int* im_tensor_data;
      im_tensor_data = (int*)(inputs[i].data.data());  // NOLINT
      in_tensor->CopyFromCpu(im_tensor_data);
    } else if (inputs[i].dtype == INT8) {
      uint8_t* im_tensor_data;
      im_tensor_data = (uint8_t*)(inputs[i].data.data());  // NOLINT
      in_tensor->CopyFromCpu(im_tensor_data);
    } else {
      std::cerr << "There's unexpected input dtype: " << inputs[i].dtype
                << std::endl;
      return false;
    }
  }
  // predict
  LOGC("Info", "before inference predictor.run()");
  predictor_->Run();
  LOGC("Info", "after inference predictor.run()");
  // get output
  auto output_names = predictor_->GetOutputNames();
  for (const auto output_name : output_names) {
    auto output_tensor = predictor_->GetOutputHandle(output_name);
    auto output_tensor_shape = output_tensor->shape();
    DataBlob output;
    output.name = output_name;
    output.shape.assign(output_tensor_shape.begin(), output_tensor_shape.end());
    output.dtype = paddle_infer::DataType(output_tensor->type());
    output.lod = output_tensor->lod();
    int size = 1;
    for (const auto &i : output_tensor_shape) {
      size *= i;
    }
    if (output.dtype == 0) {
      output.data.resize(size * sizeof(float));
      output_tensor->CopyToCpu(reinterpret_cast<float *>(output.data.data()));
    } else if (output.dtype == 1) {
      output.data.resize(size * sizeof(int64_t));
      output_tensor->CopyToCpu(reinterpret_cast<int64_t *>(output.data.data()));
    } else if (output.dtype == 2) {
      output.data.resize(size * sizeof(int));
      output_tensor->CopyToCpu(reinterpret_cast<int *>(output.data.data()));
    } else if (output.dtype == 3) {
      output.data.resize(size * sizeof(uint8_t));
      output_tensor->CopyToCpu(reinterpret_cast<uint8_t *>(output.data.data()));
    }
    outputs->push_back(std::move(output));
  }
  return true;
}

}  //  namespace PaddleDeploy
