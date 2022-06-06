//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "model_deploy/common/include/deploy_declare.h"
#include "model_deploy/common/include/base_postprocess.h"
#include "model_deploy/common/include/base_preprocess.h"
#include "model_deploy/common/include/output_struct.h"
#include "model_deploy/engine/include/engine.h"

#ifdef PADDLEX_DEPLOY_ENCRYPTION
#include "encryption/include/paddle_model_decrypt.h"
#endif  // PADDLEX_DEPLOY_ENCRYPTION

namespace PaddleDeploy {

class PD_INFER_DECL Model {
 private:
  const std::string model_type_;

 public:
  /*store the data after the YAML file has been parsed */
  YAML::Node yaml_config_;
  /* preprocess */
  std::shared_ptr<BasePreprocess> preprocess_;
  /* inference */
  std::shared_ptr<InferEngine> infer_engine_;
  /* postprocess */
  std::shared_ptr<BasePostprocess> postprocess_;

  Model() {}

  // Init model_type.
  explicit Model(const std::string model_type) : model_type_(model_type) {}

  virtual bool Init(const std::string& cfg_file, const std::string key = "") {
    if (!YamlConfigInit(cfg_file, key)) return false;
    if (!PreprocessInit()) return false;
    if (!PostprocessInit()) return false;
    return true;
  }

  virtual bool YamlConfigInit(const std::string& cfg_file,
                              const std::string key) {
    std::cerr << "Error! The Base Model was incorrectly entered" << std::endl;
    return false;
  }

  virtual bool PreprocessInit() {
    preprocess_ = nullptr;
    std::cerr << "model no Preprocess!" << std::endl;
    return false;
  }

  bool PaddleEngineInit(const PaddleEngineConfig& engine_config);

  bool TritonEngineInit(const TritonEngineConfig& engine_config);

  bool TensorRTInit(const TensorRTEngineConfig& engine_config);

  bool OpenVinoEngineInit(const OpenVinoEngineConfig& engine_config);

  virtual bool PostprocessInit() {
    postprocess_ = nullptr;
    std::cerr << "model no Postprocess!" << std::endl;
    return false;
  }

  virtual bool Predict(const std::vector<cv::Mat>& imgs,
                       std::vector<Result>* results,
                       int thread_num = 1) {
    if (!preprocess_ || !postprocess_ || !infer_engine_) {
      std::cerr << "No init,cann't predict" << std::endl;
      return false;
    }

    results->clear();
    std::vector<cv::Mat> imgs_clone;
    for (auto i = 0; i < imgs.size(); ++i) {
      imgs_clone.push_back(imgs[i].clone());
    }

    std::vector<ShapeInfo> shape_infos;
    std::vector<DataBlob> inputs;
    std::vector<DataBlob> outputs;

    if (!preprocess_->Run(&imgs_clone, &inputs, &shape_infos, thread_num)) {
      return false;
    }
    if (!infer_engine_->Infer(inputs, &outputs)) {
      return false;
    }
    if (!postprocess_->Run(outputs, shape_infos, results, thread_num)) {
      return false;
    }
    return true;
  }

  virtual bool PrePrecess(const std::vector<cv::Mat>& imgs,
                          std::vector<DataBlob>* inputs,
                          std::vector<ShapeInfo>* shape_infos,
                          int thread_num = 1) {
    if (!preprocess_) {
      std::cerr << "No PrePrecess, No pre Init. model_type=" << model_type_
                << std::endl;
      return false;
    }

    std::vector<cv::Mat> imgs_clone(imgs.size());
    for (auto i = 0; i < imgs.size(); ++i) {
      imgs[i].copyTo(imgs_clone[i]);
    }

    if (!preprocess_->Run(&imgs_clone, inputs, shape_infos, thread_num))
      return false;
    return true;
  }

  virtual void Infer(const std::vector<DataBlob>& inputs,
                     std::vector<DataBlob>* outputs) {
    infer_engine_->Infer(inputs, outputs);
  }

  virtual bool PostPrecess(const std::vector<DataBlob>& outputs,
                           const std::vector<ShapeInfo>& shape_infos,
                           std::vector<Result>* results,
                           int thread_num = 1) {
    if (!postprocess_) {
      std::cerr << "No PostPrecess, No post Init. model_type=" << model_type_
                << std::endl;
      return false;
    }
    if (postprocess_->Run(outputs, shape_infos, results, thread_num))
      return false;
    return true;
  }
};

}  // namespace PaddleDeploy
