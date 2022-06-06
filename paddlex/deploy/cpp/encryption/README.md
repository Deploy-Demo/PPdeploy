一、Linux
在 centos 7 上GCC 4.8.5 编译通过

Step1: 编译
    有两种方法, 1.1需要paddle库, 并可支持加载加密模型; 1.2不需要paddle库; 默认需要PADDLE库

    1.1 带有paddle库的编译, cmake 的命令在 build.sh 中, 请根据实际情况修改主要参数PADDLE_DIR的路径
    修改脚本设置好参数后，执行build脚本
    sh build.sh

    1.2 不带paddle库的编译, cmake 的命令在 build.sh 中, 删除-DPADDLE_DIR=${PADDLE_DIR}, 并添加
    -DWITH_PADDLE=OFF, 执行build脚本
    sh build.sh

Step2: 产出在output目录
    2.1 头文件
        include/model_code.h
        include/paddle_model_encrypt.h
        include/paddle_model_decrypt.h (注: 需要设置PADDLE库)
        include/paddle_stream_decrypt.h

    2.2 编译产出库
        lib/libpmodel-encrypt.so
        lib/libpmodel-decrypt.so (注: 此库编译需要设置PADDLE库)
        lib/libpstream-decrypt.so

    2.3 执行工具（使用-h参数查看）
        bin/paddle_encrypt_tool

二、Windows
在windows 10 Visual Studio 14 2015 上编译通过
Step1: 编译
    修改 build.bat 中 PADDLE_DIR 的路径, 以及是否需要支持PADDLE库
    执行 build.bat 脚本

Step2：打开 blend Visual Studio 2015，
    选择 open project -> 找到 Step1 中生成的 paddle—model-protect.sln -> 选择 Release 和 x64 -> ALL BUILD -> 右键生成

三、更新说明
日期: 2021-06-09
1.  sdk添加支持对流进行加密, 产出为
    libpmodel-decrypt.so 对应的头文件为paddle_model_decrypt.h, 原始的解密接口
    libpmodel-encrypt.so 对应的头文件为paddle_model_encrypt.h, 原始的加密接口, 并添加对流加密接口
    libpstream-decrypt.so 对应的头文件为paddle_stream_decrypt.h, 新的解密接口, 包含对流解密接口
