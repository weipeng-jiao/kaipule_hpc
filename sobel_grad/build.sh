#!/bin/bash

#获取当前脚本所在的目录
cur_dir=$(cd `dirname $0`; pwd)
echo cur_dir=$cur_dir


export ANDROID_NDK=/root/lib/android-ndk-r21e

export PATH=${PATH}:/opt/toolchain/7.5.0/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin

GCC_COMPILER=aarch64-linux-gnu


# build
BUILD_DIR=${cur_dir}/build

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}


cmake .. \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DCMAKE_C_COMPILER=${GCC_COMPILER}-gcc \
    -DCMAKE_CXX_COMPILER=${GCC_COMPILER}-g++     
    # -DCMAKE_BUILD_TYPE=Debug


# cmake   .. \
#         -DCMAKE_SYSTEM_NAME=Android \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
#         -DANDROID_ABI="arm64-v8a" \
#         -DANDROID_NDK=${ANDROID_NDK} \
#         -DPLATFORM=${SYSTEM_PLATFORM} \
#         -DANDROID_ARM_NEON=ON \
#         -DANDROID_PLATFORM=android-23 \
#         -DOpenCV_DIR=./../opencv/android/4.0.0/arm64-v8a/sdk/native/jni \
#         -DCMAKE_INSTALL_PREFIX=../install 
        
make -j8
make install


