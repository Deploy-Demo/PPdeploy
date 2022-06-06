# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=ON
# 使用MKL or openblas
WITH_MKL=OFF
# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_PADDLE_TENSORRT=OFF
# TensorRT 的路径，如果需要集成TensorRT，需修改为您实际安装的TensorRT路径
TENSORRT_DIR=$(pwd)/TensorRT/
# Paddle 预测库路径, 请修改为您实际安装的预测库路径
PADDLE_DIR=$(pwd)/paddle_inference
# Paddle 的预测库是否使用静态库来编译
# 使用TensorRT时，Paddle的预测库通常为动态库
WITH_STATIC_LIB=OFF
# CUDA 的 lib 路径
CUDA_LIB=/usr/local/cuda/lib64
# CUDNN 的 lib 路径
CUDNN_LIB=/usr/lib/aarch64-linux-gnu
# 是否加密
WITH_ENCRYPTION=OFF
# OPENSSL 路径
OPENSSL_DIR=$(pwd)/deps/openssl-1.1.0k


# 以下无需改动
rm -rf build
mkdir -p build
cd build
sudo cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_PADDLE_TENSORRT=${WITH_PADDLE_TENSORRT} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DWITH_ENCRYPTION=${WITH_ENCRYPTION} \
    -DOPENSSL_DIR=${OPENSSL_DIR}
make -j8
