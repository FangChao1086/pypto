export python3=/data/g00895580/miniconda3/bin/python3.12
export WORKSPACE_DIR=/data/g00895580/ptoas/
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

export LLVM_SOURCE_DIR=/data/g00895580/mlir/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)
export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH

pip3 install -e .
source /data/g00895580/Ascend/cann/bin/setenv.bash
source /data/g00895580/Ascend/cann-8.5.0/set_env.sh
source /data/g00895580/Ascend/ascend-toolkit/set_env.sh

