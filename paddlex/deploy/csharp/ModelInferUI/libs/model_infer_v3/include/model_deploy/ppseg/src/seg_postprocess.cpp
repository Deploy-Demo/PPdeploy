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

#include "model_deploy/ppseg/include/seg_postprocess.h"

#include <time.h>

namespace PaddleDeploy {

bool SegPostprocess::Init(const YAML::Node& yaml_config) {
  if (yaml_config["version"].IsDefined() &&
      yaml_config["toolkit"].as<std::string>() == "PaddleX") {
    version_ = yaml_config["version"].as<std::string>();
  } else {
    version_ = "0.0.0";
  }
  return true;
}

void SegPostprocess::RestoreSegMap(const ShapeInfo& shape_info,
                                   cv::Mat* label_mat, cv::Mat* score_mat,
                                   SegResult* result) {
  int ori_h = shape_info.shapes[0][1];
  int ori_w = shape_info.shapes[0][0];
  int score_c = score_mat->channels();
  result->label_map.Resize({ori_h, ori_w});
  result->score_map.Resize({ori_h, ori_w, score_c});

  for (int j = shape_info.transforms.size() - 1; j > 0; --j) {
    std::vector<int> last_shape = shape_info.shapes[j - 1];
    std::vector<int> cur_shape = shape_info.shapes[j];
    if (shape_info.transforms[j] == "Resize" ||
        shape_info.transforms[j] == "ResizeByShort" ||
        shape_info.transforms[j] == "ResizeByLong") {
      if (last_shape[0] != label_mat->cols ||
          last_shape[1] != label_mat->rows) {
        cv::resize(*label_mat, *label_mat,
                   cv::Size(last_shape[0], last_shape[1]), 0, 0,
                   cv::INTER_NEAREST);
        cv::resize(*score_mat, *score_mat,
                   cv::Size(last_shape[0], last_shape[1]), 0, 0,
                   cv::INTER_LINEAR);
      }
    } else if (shape_info.transforms[j] == "Padding") {
      if (last_shape[0] < label_mat->cols || last_shape[1] < label_mat->rows) {
        *label_mat = (*label_mat)(cv::Rect(0, 0, last_shape[0], last_shape[1]));
        *score_mat = (*score_mat)(cv::Rect(0, 0, last_shape[0], last_shape[1]));
      }
    }
  }
  if (label_mat->isContinuous()) {
    result->label_map.data.assign(
        reinterpret_cast<const uint8_t*>(label_mat->data),
        reinterpret_cast<const uint8_t*>(label_mat->data) +
            label_mat->total() * (label_mat->channels()));
  } else {
    for (int i = 0; i < label_mat->rows; ++i) {
      result->label_map.data.insert(
          result->label_map.data.end(), label_mat->ptr<uint8_t>(i),
          label_mat->ptr<uint8_t>(i) +
              label_mat->cols * (label_mat->channels()));
    }
  }

  if (score_mat->isContinuous()) {
    result->score_map.data.assign(
        reinterpret_cast<const float*>(score_mat->data),
        reinterpret_cast<const float*>(score_mat->data) +
            score_mat->total() * (score_mat->channels()));
  } else {
    for (int i = 0; i < score_mat->rows; ++i) {
      result->score_map.data.insert(
          result->score_map.data.end(), score_mat->ptr<float>(i),
          score_mat->ptr<float>(i) + score_mat->cols * (score_mat->channels()));
    }
  }
}

// ppseg version >= 2.1  shape = [b, w, h]
bool SegPostprocess::RunV2(const DataBlob& output,
                           const std::vector<ShapeInfo>& shape_infos,
                           std::vector<Result>* results, int thread_num) {
  int batch_size = shape_infos.size();
  int label_map_size = output.shape[1] * output.shape[2];
  const uint8_t* label_data;
  std::vector<uint8_t> label_vector;
  if (output.dtype == INT64) {  // int64
    const int64_t* output_data =
        reinterpret_cast<const int64_t*>(output.data.data());
    std::transform(output_data, output_data + label_map_size * batch_size,
                   std::back_inserter(label_vector),
                   [](int64_t x) { return (uint8_t)x; });
    label_data = reinterpret_cast<const uint8_t*>(label_vector.data());
  } else if (output.dtype == INT32) {  // int32
    const int32_t* output_data =
        reinterpret_cast<const int32_t*>(output.data.data());
    std::transform(output_data, output_data + label_map_size * batch_size,
                   std::back_inserter(label_vector),
                   [](int32_t x) { return (uint8_t)x; });
    label_data = reinterpret_cast<const uint8_t*>(label_vector.data());
  } else if (output.dtype == INT8) {  // uint8
    label_data = reinterpret_cast<const uint8_t*>(output.data.data());
  } else {
    std::cerr << "Output dtype is not support on seg posrtprocess "
              << output.dtype << std::endl;
    return false;
  }

  for (int i = 0; i < batch_size; ++i) {
    (*results)[i].model_type = "seg";
    (*results)[i].seg_result = new SegResult();
    const uint8_t* current_start_ptr = label_data + i * label_map_size;
    cv::Mat score_mat(output.shape[1], output.shape[2], CV_32FC1,
                      cv::Scalar(1.0));
    cv::Mat label_mat(output.shape[1], output.shape[2], CV_8UC1,
                      const_cast<uint8_t*>(current_start_ptr));

    RestoreSegMap(shape_infos[i], &label_mat, &score_mat,
                  (*results)[i].seg_result);
  }
  return true;
}

// paddlex version >= 2.0.0 shape = [b, h, w, c]
bool SegPostprocess::RunXV2(const std::vector<DataBlob>& outputs,
                            const std::vector<ShapeInfo>& shape_infos,
                            std::vector<Result>* results, int thread_num) {
  int batch_size = shape_infos.size();
  int label_map_size = outputs[0].shape[1] * outputs[0].shape[2];
  std::vector<int> score_map_shape = outputs[1].shape;
  int score_map_size =
      std::accumulate(score_map_shape.begin() + 1, score_map_shape.end(), 1,
                      std::multiplies<int>());
  const uint8_t* label_map_data;
  std::vector<uint8_t> label_map_vector;
  if (outputs[0].dtype == INT32) {
    const int32_t* output_data =
        reinterpret_cast<const int32_t*>(outputs[0].data.data());
    std::transform(output_data, output_data + label_map_size * batch_size,
                   std::back_inserter(label_map_vector),
                   [](int32_t x) { return (uint8_t)x; });
    label_map_data = reinterpret_cast<const uint8_t*>(label_map_vector.data());
  }
  const float* score_map_data =
      reinterpret_cast<const float*>(outputs[1].data.data());
  for (int i = 0; i < batch_size; ++i) {
    (*results)[i].model_type = "seg";
    (*results)[i].seg_result = new SegResult();
    const uint8_t* current_label_start_ptr =
        label_map_data + i * label_map_size;
    const float* current_score_start_ptr = score_map_data + i * score_map_size;
    cv::Mat label_mat(outputs[0].shape[1], outputs[0].shape[2], CV_8UC1,
                      const_cast<uint8_t*>(current_label_start_ptr));
    cv::Mat score_mat(score_map_shape[1], score_map_shape[2],
                      CV_32FC(score_map_shape[3]),
                      const_cast<float*>(current_score_start_ptr));
    RestoreSegMap(shape_infos[i], &label_mat, &score_mat,
                  (*results)[i].seg_result);
  }
  return true;
}

bool SegPostprocess::Run(const std::vector<DataBlob>& outputs,
                         const std::vector<ShapeInfo>& shape_infos,
                         std::vector<Result>* results, int thread_num) {
  if (outputs.size() == 0) {
    std::cerr << "empty output on SegPostprocess" << std::endl;
    return true;
  }
  results->clear();
  int batch_size = shape_infos.size();
  results->resize(batch_size);

  // tricks for PaddleX, of which segmentation model has two outputs
  int index = 0;
  if (outputs.size() == 2) {
    index = 1;
  }
  std::vector<int> score_map_shape = outputs[index].shape;
  // paddlex version >= 2.0.0 shape[b, h, w, c]
  if (version_ >= "2.0.0") {
    return RunXV2(outputs, shape_infos, results, thread_num);
  }
  // ppseg version >= 2.1  shape = [b, h, w]
  if (score_map_shape.size() == 3) {
    return RunV2(outputs[index], shape_infos, results, thread_num);
  }

  int score_map_size =
      std::accumulate(score_map_shape.begin() + 1, score_map_shape.end(), 1,
                      std::multiplies<int>());
  const float* score_map_data =
      reinterpret_cast<const float*>(outputs[index].data.data());
  int num_map_pixels = score_map_shape[2] * score_map_shape[3];

  for (int i = 0; i < batch_size; ++i) {
    (*results)[i].model_type = "seg";
    (*results)[i].seg_result = new SegResult();
    const float* current_start_ptr = score_map_data + i * score_map_size;
    cv::Mat ori_score_mat(score_map_shape[1],
                          score_map_shape[2] * score_map_shape[3], CV_32FC1,
                          const_cast<float*>(current_start_ptr));
    ori_score_mat = ori_score_mat.t();
    cv::Mat score_mat(score_map_shape[2], score_map_shape[3], CV_32FC1);
    cv::Mat label_mat(score_map_shape[2], score_map_shape[3], CV_8UC1);
    for (int j = 0; j < ori_score_mat.rows; ++j) {
      double max_value;
      cv::Point max_id;
      minMaxLoc(ori_score_mat.row(j), 0, &max_value, 0, &max_id);
      score_mat.at<float>(j) = max_value;
      label_mat.at<uchar>(j) = max_id.x;
    }
    RestoreSegMap(shape_infos[i], &label_mat, &score_mat,
                  (*results)[i].seg_result);
  }
  return true;
}

}  //  namespace PaddleDeploy
