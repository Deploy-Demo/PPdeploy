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

#include "model_deploy/ppdet/include/det_postprocess.h"

namespace PaddleDeploy {

bool DetPostprocess::Init(const YAML::Node& yaml_config) {
  labels_.clear();
  for (auto item : yaml_config["labels"]) {
    std::string label = item.as<std::string>();
    labels_.push_back(label);
  }
  version_ = yaml_config["version"].as<std::string>();
  return true;
}

bool DetPostprocess::ProcessBbox(const std::vector<DataBlob>& outputs,
                                 const std::vector<ShapeInfo>& shape_infos,
                                 std::vector<Result>* results, int thread_num) {
  const float* data = reinterpret_cast<const float*>(outputs[0].data.data());

  std::vector<int> num_bboxes_each_sample;
  if (outputs[0].lod.empty()) {
    for (auto i = 0; i < shape_infos.size(); ++i) {
      num_bboxes_each_sample.push_back(
          outputs[0].shape[0]/static_cast<int>(shape_infos.size()));
    }
  } else {
    for (auto i = 0; i < outputs[0].lod[0].size() - 1; ++i) {
      int num = outputs[0].lod[0][i + 1] - outputs[0].lod[0][i];
      num_bboxes_each_sample.push_back(num);
    }
  }

  int idx = 0;
  for (auto i = 0; i < num_bboxes_each_sample.size(); ++i) {
    (*results)[i].model_type = "det";
    (*results)[i].det_result = new DetResult();
    for (auto j = 0; j < num_bboxes_each_sample[i]; ++j) {
      Box box;
      box.category_id = static_cast<int>(round(data[idx * 6]));
      if (box.category_id < 0) {
        std::cerr << "Compute category id is less than 0"
                  << "(Maybe no object detected)" << std::endl;
        return true;
      }
      if (box.category_id >= labels_.size()) {
        std::cerr << "Compute category id is greater than labels "
                  << "in your config file" << std::endl;
        std::cerr << "Compute Category ID: " << box.category_id
                  << ", but length of labels is " << labels_.size()
                  << std::endl;
        return false;
      }
      box.category = labels_[box.category_id];
      box.score = data[idx * 6 + 1];
      // TODO(jiangjiajun): only for RCNN and YOLO
      // lack of process for SSD and Face
      float xmin = data[idx * 6 + 2];
      float ymin = data[idx * 6 + 3];
      float xmax = data[idx * 6 + 4];
      float ymax = data[idx * 6 + 5];
      box.coordinate = {xmin, ymin, xmax - xmin, ymax - ymin};
      (*results)[i].det_result->boxes.push_back(std::move(box));
      idx += 1;
    }
  }
  return true;
}

bool DetPostprocess::ProcessMask(DataBlob* mask_blob,
                                 const std::vector<ShapeInfo>& shape_infos,
                                 std::vector<Result>* results,
                                 float threshold) {
  std::vector<int> output_mask_shape = mask_blob->shape;
  float *mask_data = reinterpret_cast<float*>(mask_blob->data.data());
  int mask_pixels = output_mask_shape[2] * output_mask_shape[3];
  int classes = output_mask_shape[1];
  for (auto i = 0; i < results->size(); ++i) {
    (*results)[i].det_result->mask_resolution = output_mask_shape[2];
    for (auto j = 0; j < (*results)[i].det_result->boxes.size(); ++j) {
      Box *box = &(*results)[i].det_result->boxes[j];
      auto begin_mask_data = mask_data + box->category_id * mask_pixels;
      cv::Mat bin_mask(output_mask_shape[2],
                       output_mask_shape[3],
                       CV_32FC1,
                       begin_mask_data);
      // expand box
      cv::Scalar value = cv::Scalar(0.0);
      cv::copyMakeBorder(bin_mask, bin_mask,
                         1, 1, 1, 1,
                         cv::BORDER_CONSTANT,
                         value = value);

      int max_w = shape_infos[i].shapes[0][0];
      int max_h = shape_infos[i].shapes[0][1];
      double scale = (output_mask_shape[2] + 2.0) / output_mask_shape[2];
      double w_half = static_cast<double>(box->coordinate[2]) * 0.5;
      double h_half = static_cast<double>(box->coordinate[3]) * 0.5;
      double x_c = static_cast<double>(box->coordinate[0]) + w_half;
      double y_c = static_cast<double>(box->coordinate[1]) + h_half;
      w_half *= scale;
      h_half *= scale;
      int x_min = static_cast<int>(x_c - w_half);
      int x_max = static_cast<int>(x_c + w_half);
      int y_min = static_cast<int>(y_c - h_half);
      int y_max = static_cast<int>(y_c + h_half);

      cv::resize(bin_mask, bin_mask,
                 cv::Size(std::max(x_max - x_min + 1, 1),
                          std::max(y_max - y_min + 1, 1)));

      cv::threshold(bin_mask, bin_mask, threshold, 1, cv::THRESH_BINARY);
      bin_mask.convertTo(bin_mask, CV_8UC1);

      int x0 = std::min(std::max(x_min, 0), max_w);
      int x1 = std::min(std::max(x_max + 1, 0), max_w);
      int y0 = std::min(std::max(y_min, 0), max_h);
      int y1 = std::min(std::max(y_max + 1, 0), max_h);

      cv::Mat mask_mat = bin_mask(cv::Range(y0 - y_min, y1 - y_min),
                                  cv::Range(x0 - x_min, x1 - x_min));
      // expand image
      cv::copyMakeBorder(mask_mat, mask_mat,
                         max_h - y1,
                         y0,
                         x0,
                         max_w - x1,
                         cv::BORDER_CONSTANT,
                         value = value);

      box->mask.Clear();
      box->mask.shape = {max_h, max_w};
      if (mask_mat.isContinuous()) {
        box->mask.data.assign(mask_mat.datastart, mask_mat.dataend);
      } else {
        for (auto i = 0; i < mask_mat.rows; ++i) {
          box->mask.data.insert(box->mask.data.end(),
                                mask_mat.ptr<uint8_t>(i),
                                mask_mat.ptr<uint8_t>(i) + mask_mat.cols);
        }
      }
      mask_data += classes * mask_pixels;
    }
  }
  return true;
}

bool DetPostprocess::ProcessMaskV2(DataBlob* mask_blob,
                                 const std::vector<ShapeInfo>& shape_infos,
                                 std::vector<Result>* results) {
  std::vector<int> output_mask_shape = mask_blob->shape;
  float *mask_data = reinterpret_cast<float*>(mask_blob->data.data());
  int mask_pixels = output_mask_shape[1] * output_mask_shape[2];
  for (auto i = 0; i < results->size(); ++i) {
    for (auto j = 0; j < (*results)[i].det_result->boxes.size(); ++j) {
      Box *box = &(*results)[i].det_result->boxes[j];

      auto begin_mask = mask_data + j * mask_pixels;
      cv::Mat bin_mask(output_mask_shape[1],
                       output_mask_shape[2],
                       CV_32SC1,
                       begin_mask);
      bin_mask.convertTo(bin_mask, CV_8UC1);

      box->mask.Clear();
      box->mask.shape = {static_cast<int>(output_mask_shape[1]),
                      static_cast<int>(output_mask_shape[2])};
      if (bin_mask.isContinuous()) {
        box->mask.data.assign(bin_mask.datastart, bin_mask.dataend);
      } else {
        for (auto i = 0; i < bin_mask.rows; ++i) {
          box->mask.data.insert(box->mask.data.end(),
                                bin_mask.ptr<uint8_t>(i),
                                bin_mask.ptr<uint8_t>(i) + bin_mask.cols);
        }
      }
    }
  }
  return true;
}

bool DetPostprocess::Run(const std::vector<DataBlob>& outputs,
                         const std::vector<ShapeInfo>& shape_infos,
                         std::vector<Result>* results, int thread_num) {
  results->clear();
  if (outputs.size() == 0) {
    std::cerr << "empty output on DetPostprocess" << std::endl;
    return true;
  }
  results->resize(shape_infos.size());
  if (!ProcessBbox(outputs, shape_infos, results, thread_num)) {
    std::cerr << "Error happend while process bboxes" << std::endl;
    return false;
  }

  if (version_ < "2.0" && outputs.size() == 2) {
    DataBlob mask_blob = outputs[1];
    if (!ProcessMask(&mask_blob, shape_infos, results)) {
      std::cerr << "Error happend while process masks" << std::endl;
      return false;
    }
  } else if (version_ >= "2.0" && outputs.size() == 3) {
    DataBlob mask_blob = outputs[2];
    if (!ProcessMaskV2(&mask_blob, shape_infos, results)) {
      std::cerr << "Error happend while process masks" << std::endl;
      return false;
    }
  }
  return true;
}

}  //  namespace PaddleDeploy
