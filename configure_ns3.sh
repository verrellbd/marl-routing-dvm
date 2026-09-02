#!/bin/bash
# ns-3 configure with the flags this project needs (pybind11, protobuf, venv python).
# ns-3.42 + ns3-ai commit b8c9858
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_DIR="${NS3_DIR:-$REPO/ns-3-dev}"
# protobuf came from an Anaconda install here; override if yours is elsewhere
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-/opt/anaconda3}"
cd "$NS3_DIR" || exit 1
PYBIND11_CMAKE_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
VENV_PYTHON=$(which python)
rm -rf cmake-cache build .lock-ns3_linux_build
./ns3 configure --enable-examples --enable-tests -- \
    -DPython_EXECUTABLE=$VENV_PYTHON \
    -Dpybind11_DIR=$PYBIND11_CMAKE_DIR \
    -DProtobuf_DIR=$CONDA_PREFIX_DIR/lib/cmake/protobuf \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX_DIR
