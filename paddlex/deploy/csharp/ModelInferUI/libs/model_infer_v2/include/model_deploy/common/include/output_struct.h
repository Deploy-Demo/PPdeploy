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

#include <algorithm>
#include <iostream>
#include <ostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <cassert>
#include <functional>
#include <numeric>
#include <iomanip>

namespace PaddleDeploy {

enum Dtype {FLOAT32, INT64, INT32, INT8};

struct DataBlob {
  // data
  std::vector<char> data;

  // data name
  std::string name;

  // data shape
  std::vector<int> shape;

  /*
    data dtype
    0: FLOAT32
    1: INT64
    2: INT32
    3: UINT8
    */
  int dtype;

  // Lod Info
  std::vector<std::vector<size_t>> lod;

  DataBlob() {}
  explicit DataBlob(const std::string& blob_name) {
    name = blob_name;
  }
  void Resize(const std::vector<int>& new_shape, int data_type) {
    assert(data_type >= 0 && data_type < 4);
    int unit = 4;
    if (data_type == INT64) {
      unit = 8;
    } else if (data_type == INT8) {
      unit = 1;
    }
    int total_size = std::accumulate(new_shape.begin(),
                new_shape.end(), 1, std::multiplies<int>());
    data.resize(total_size * unit);
    shape.clear();
    shape.assign(new_shape.begin(), new_shape.end());
    dtype = data_type;
  }
};

struct ShapeInfo {
  std::vector<std::vector<int>> shapes;
  std::vector<std::string> transforms;
  bool has_batch_padding = false;
  void Insert(const std::string& name, int width, int height) {
    transforms.push_back(name);
    shapes.push_back({width, height});
  }
};

template<typename T>
void cal_mean_std(const std::vector<T>& values, double* mean, double* std) {
  *mean = 0;
  *std = 0;
  for (auto i = 0; i < values.size(); ++i) {
    (*mean) += static_cast<double>(values[i]);
  }
  (*mean) = (*mean) / static_cast<double>(values.size());

  if (values.size() < 2) {
    *std = 0.0;
  } else {
    double tmp = 0.0;
    for (auto i = 0; i < values.size(); ++i) {
      tmp += (values[i] - (*mean)) * (values[i] - (*mean));
    }
    *std = tmp / (values.size() - 1);
  }
}

template <class T>
struct Mask {
  // raw data of mask
  std::vector<T> data;
  // the shape of mask
  std::vector<int> shape;
  void Resize(const std::vector<int>& new_shape) {
    int total_size = std::accumulate(new_shape.begin(),
                new_shape.end(), 1, std::multiplies<int>());
    data.resize(total_size);
    shape.clear();
    shape.assign(new_shape.begin(), new_shape.end());
  }
  void Clear() {
    data.clear();
    shape.clear();
  }
  friend std::ostream &operator<<(std::ostream & stream, const Mask<T>& mask) {
    double m;
    double d;
    cal_mean_std<T>(mask.data, &m, &d);
    stream << "Mask(mean:\t" << m << "\tstd:\t" << d << ")";
    return stream;
  }
};

struct Box {
  int category_id;
  // category label this box belongs to
  std::string category;
  // confidence score
  float score;
  std::vector<float> coordinate;
  Mask<uint8_t> mask;

  friend std::ostream &operator<<(std::ostream & stream, const Box& b) {
    stream << "Box(" << b.category_id << "\t" << b.category << "\t" << b.score;
    for (auto i = 0; i < b.coordinate.size(); ++i) {
        stream << "\t" << b.coordinate[i];
    }
    stream << ")";
    if (b.mask.data.size() != 0) {
      stream << "\n" << b.mask;
      for (auto shape : b.mask.shape) {
        stream << "\t" << shape;
      }
    }
    return stream;
  }
};

struct ClasResult {
  // target boxes
  int category_id;
  std::string category;
  double score;
  friend std::ostream &operator<<(std::ostream & stream, const ClasResult& c) {
    stream << "Classify(" << c.category_id << "\t"
           << c.category << "\t" << c.score << ")";
    return stream;
  }
};

struct DetResult {
  // target boxes
  std::vector<Box> boxes;
  int mask_resolution;
  void clear() {
    boxes.clear();
  }
  friend std::ostream &operator<<(std::ostream & stream, const DetResult& d) {
    for (auto i = 0; i < d.boxes.size(); ++i) {
      stream << d.boxes[i];
      if (i != d.boxes.size() - 1) {
        stream << "\n";
      }
    }
    return stream;
  }
};

struct OcrResult {
  std::vector<std::vector<std::vector<int>>> boxes;
  float cls_score;
  float crnn_score;
  int label;
  std::vector<std::string> str_res;
};

struct SegResult {
  // represent label of each pixel on image matrix
  Mask<uint8_t> label_map;
  // represent score of each pixel on image matrix
  Mask<float> score_map;
  friend std::ostream &operator<<(std::ostream & stream, const SegResult& s) {
    stream << "Score" << s.score_map << "\tLabel" << s.label_map;
    return stream;
  }
};

struct Result {
  std::string model_type;
  union {
    ClasResult* clas_result;
    DetResult* det_result;
    SegResult* seg_result;
    OcrResult* ocr_result;
  };

  Result() {
    clas_result = nullptr;
  }

  explicit Result(std::string result_type) {
    model_type = result_type;
    clas_result = nullptr;
  }

  friend std::ostream &operator<<(std::ostream & stream, const Result& r) {
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(8);
    if ("det" == r.model_type) {
      if (nullptr == r.det_result) {
        stream << "det_result is not initialized";
      } else {
        stream << *(r.det_result);
      }
    } else if ("clas" == r.model_type) {
      if (nullptr == r.clas_result) {
        stream << "clas_result is not initialized";
      } else {
        stream << *(r.clas_result);
      }
    } else if ("seg" == r.model_type) {
      stream << *(r.seg_result);
    } else {
      stream << "Result is not support to print";
    }
    return stream;
  }

  void Clear() {
    if ("det" == model_type) {
      delete det_result;
      det_result = NULL;
    } else if ("seg" == model_type) {
      delete seg_result;
      seg_result = NULL;
    } else if ("clas" == model_type) {
      delete clas_result;
      clas_result = NULL;
    } else if ("ocr" == model_type) {
      delete ocr_result;
      ocr_result = NULL;
    }
  }

  Result(const Result& result) {
    Clear();
    model_type = result.model_type;
    if ("det" == model_type) {
      det_result = new DetResult();
      *det_result = *(result.det_result);
    } else if ("seg" == model_type) {
      seg_result = new SegResult();
      *seg_result = *(result.seg_result);
    } else if ("clas" == model_type) {
      clas_result = new ClasResult();
      *clas_result = *(result.clas_result);
    } else if ("ocr" == model_type) {
      ocr_result = new OcrResult();
      *ocr_result = *(result.ocr_result);
    }
  }

  Result& operator=(const Result& result) {
    Clear();
    model_type = result.model_type;
    if ("det" == model_type) {
      det_result = new DetResult();
      *det_result = *(result.det_result);
    } else if ("seg" == model_type) {
      seg_result = new SegResult();
      *seg_result = *(result.seg_result);
    } else if ("clas" == model_type) {
      clas_result = new ClasResult();
      *clas_result = *(result.clas_result);
    } else if ("ocr" == model_type) {
      ocr_result = new OcrResult();
      *ocr_result = *(result.ocr_result);
    }
    return *this;
  }


  ~Result() {
    Clear();
  }
};

}  // namespace PaddleDeploy
