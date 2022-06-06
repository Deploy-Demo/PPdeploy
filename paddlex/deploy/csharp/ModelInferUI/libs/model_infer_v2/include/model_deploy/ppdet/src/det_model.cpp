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
#include "model_deploy/ppdet/include/det_model.h"
#include "model_deploy/ppdet/include/det_standard_config.h"

namespace PaddleDeploy {

bool DetModel::GenerateTransformsConfig(const YAML::Node& src) {
  assert(src["Preprocess"].IsDefined());
  assert(src["arch"].IsDefined());
  std::string model_arch = src["arch"].as<std::string>();
  yaml_config_["transforms"]["BGR2RGB"] = YAML::Null;
  for (const auto& op : src["Preprocess"]) {
    assert(op["type"].IsDefined());
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "Normalize") {
      DetNormalize(op, &yaml_config_);
    } else if (op_name == "NormalizeImage") {
      DetNormalize(op, &yaml_config_);
    } else if (op_name == "Permute") {
      DetPermute(op, &yaml_config_);
    } else if (op_name == "Resize") {
      DetResize(op, &yaml_config_, model_arch);
    } else if (op_name == "PadStride") {
      DetPadStride(op, &yaml_config_);
    } else {
      std::cerr << "Unexpected transforms op name: '"
                << op_name << "'" << std::endl;
      return false;
    }
  }
  return true;
}

bool DetModel::YamlConfigInit(const std::string& cfg_file,
                              const std::string key) {
  YAML::Node det_config;
  if ("" == key) {
    det_config = YAML::LoadFile(cfg_file);
  } else {
#ifdef PADDLEX_DEPLOY_ENCRYPTION
    std::string cfg = decrypt_file(cfg_file.c_str(), key.c_str());
    det_config = YAML::Load(cfg);
#else
     std::cerr << "Don't open encryption on compile" << std::endl;
    return false;
#endif  // PADDLEX_DEPLOY_ENCRYPTION
  }

  yaml_config_["model_format"] = "Paddle";
  // arch support value:YOLO, SSD, RetinaNet, RCNN, Face
  if (!det_config["arch"].IsDefined()) {
    std::cerr << "Fail to find arch in PaddleDection yaml file" << std::endl;
    return false;
  } else if (!det_config["label_list"].IsDefined()) {
    std::cerr << "Fail to find label_list in "
              << "PaddleDection yaml file"
              << std::endl;
    return false;
  }
  yaml_config_["model_name"] = det_config["arch"].as<std::string>();
  yaml_config_["toolkit"] = "PaddleDetection";
  if (det_config["version"].IsDefined()) {
    yaml_config_["version"] = det_config["version"].as<std::string>();
  } else if (det_config["use_python_inference"].IsDefined()) {
    yaml_config_["version"] = "0.5";
  } else if (!det_config["use_python_inference"].IsDefined()) {
    yaml_config_["version"] = "2.0";
  }

  int i = 0;
  for (const auto& label : det_config["label_list"]) {
    yaml_config_["labels"][i] = label.as<std::string>();
    i++;
  }

  // Generate Standard Transforms Configuration
  if (!GenerateTransformsConfig(det_config)) {
    std::cerr << "Fail to generate standard configuration "
              << "of tranforms" << std::endl;
    return false;
  }
  return true;
}

bool DetModel::PreprocessInit() {
  preprocess_ = std::make_shared<DetPreprocess>();
  if (!preprocess_->Init(yaml_config_))
    return false;
  return true;
}

bool DetModel::PostprocessInit() {
  postprocess_ = std::make_shared<DetPostprocess>();
  if (!postprocess_->Init(yaml_config_))
    return false;
  return true;
}

}  // namespace PaddleDeploy
