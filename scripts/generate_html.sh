#!/bin/bash

# http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

#---
set -ex

PROJ_DIR="$(dirname "${DIR}")"
cd ${PROJ_DIR}

MINICONDA="$HOME/miniconda"
export PATH="${MINICONDA}/bin:$PATH"
hash -r

source activate dshinpy

#---
EXCLUDE_DIRS=(
./dshin/third_party
)

GEN_PATH=./docs/apidoc

#---
# Remove directory if exists.
if [ -d ${GEN_PATH} ] ; then
    rm -r ${GEN_PATH}
fi

mkdir -p ${GEN_PATH}

sphinx-apidoc \
    -o ${GEN_PATH} \
    ./dshin \
    ${EXCLUDE_DIRS[@]}

cd ${PROJ_DIR}/docs

make html

make singlehtml
