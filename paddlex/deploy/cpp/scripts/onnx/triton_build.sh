#!/bin/bash

for i in "$@"; do
    case $i in
        --triton_client=*)
         TRITON_CLIENT="${i#*=}"
         shift
         ;;
        *)
         echo "unknown option $i"
         exit 1
         ;;
    esac
done

if [ $TRITON_CLIENT ];then
	echo "TRITON_CLIENT = $TRITON_CLIENT"
else
	echo "TRITON_CLIENT is not exist, please set by --triton_client"
    exit 1
fi

# install relayed library
sh $(pwd)/scripts/triton_env.sh

# download opencv library
OPENCV_DIR=$(pwd)/deps/opencv3.4.6gcc4.8ffmpeg/
{
    bash $(pwd)/scripts/bootstrap.sh ${OPENCV_DIR}
} || {
    echo "Fail to execute script/bootstrap.sh"
    exit -1
}

# download glog library
GLOG_DIR=$(pwd)/deps/glog/
GLOG_URL=https://bj.bcebos.com/paddlex/deploy/glog.tar.gz

if [ ! -d $(pwd)deps/ ]; then
    mkdir -p deps
fi

if [ ! -d ${GLOG_DIR} ]; then
    cd deps
    wget -c ${GLOG_URL} -O glog.tar.gz
    tar -zxvf glog.tar.gz
    rm -rf glog.tar.gz
    cd ..
fi

# download gflogs library
GFLAGS_DIR=$(pwd)/deps/gflags/
GFLAGS_URL=https://bj.bcebos.com/paddlex/deploy/gflags.tar.gz
if [ ! -d ${GFLAGS_DIR} ]; then
    cd deps
    wget -c ${GFLAGS_URL} -O glags.tar.gz
    tar -zxvf gflags.tar.gz
    rm -rf gflags.tar.gz
    cd ..
fi

# install libpng needed by opencv
ldconfig -p | grep png16 > log
if [ $? -ne 0 ];then
    apt-get install libpng16-16
fi

# install libjasper1 needed by opencv
ldconfig -p | grep libjasper  > log
if [  $? -ne 0  ];then
    add-apt-repository > log
    if [  $? -ne 0 ]; then
        apt-get install software-properties-common
    fi
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
    apt update
    apt install libjasper1 libjasper-dev
fi

rm -rf build
mkdir -p build
cd build
cmake ../demo/onnx_triton/ \
    -DTRITON_CLIENT=${TRITON_CLIENT} \
    -DOPENCV_DIR=${OPENCV_DIR}  \
    -DGLOG_DIR=${GLOG_DIR} \
    -DGFLAGS_DIR=${GFLAGS_DIR}

make -j16
