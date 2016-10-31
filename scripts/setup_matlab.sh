#!/usr/bin/env bash

[ -z "$MATLAB_ROOT" ] && echo "Need to set MATLAB_ROOT" && exit 1;
[ -z "$CONDA_ENV" ] && echo "Need to set CONDA_ENV" && exit 1;

set -ex

cd ${MATLAB_ROOT}/extern/engines/python
source activate ${CONDA_ENV}


BUILD_DIR="/tmp/matlab_python_build"

if [[ -d ${BUILD_DIR} ]]; then
    echo "Removing ${BUILD_DIR}"
    rm -r ${BUILD_DIR}
fi

python setup.py install
