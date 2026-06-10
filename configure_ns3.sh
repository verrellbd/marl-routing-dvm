#!/bin/bash
# Working ns-3 configure for malmo cluster - verified May 21, 2026
# ns-3.42 + ns3-ai commit b8c9858
cd ~/thesis/ns-3-dev
PYBIND11_CMAKE_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
VENV_PYTHON=$(which python)
rm -rf cmake-cache build .lock-ns3_linux_build
./ns3 configure --enable-examples --enable-tests -- \
    -DPython_EXECUTABLE=$VENV_PYTHON \
    -Dpybind11_DIR=$PYBIND11_CMAKE_DIR \
    -DProtobuf_DIR=/opt/anaconda3/lib/cmake/protobuf \
    -DCMAKE_PREFIX_PATH=/opt/anaconda3
