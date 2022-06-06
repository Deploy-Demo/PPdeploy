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
#pragma once

#include <memory>
#include <string>

#include "model_deploy/common/include/deploy_declare.h"
#include "model_deploy/common/include/output_struct.h"
#include "model_deploy/common/include/model_factory.h"
#include "model_deploy/engine/include/engine.h"

#ifdef PADDLEX_DEPLOY_ENCRYPTION
#include "encryption/include/paddle_model_encrypt.h"
#endif  // PADDLEX_DEPLOY_ENCRYPTION

namespace PaddleDeploy {

PD_INFER_DECL Model* CreateModel(const std::string& name);

}  // namespace PaddleDeploy
