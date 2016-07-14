#!/usr/bin/env bash

PYTHON_VERSION=3.5
MINICONDA="$HOME/miniconda"

### 
# http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
PROJ_DIR=${DIR}/../
### 

set -ex


if [ ! -d ${MINICONDA} ]; then
    if [[ "$PYTHON_VERSION" == 2* ]]; then
        WGET_MINICONDA="https://repo.continuum.io/miniconda/Miniconda2-latest-"
    else
        WGET_MINICONDA="https://repo.continuum.io/miniconda/Miniconda3-latest-"
    fi


    if [ "$(uname)" == "Darwin" ]; then
        # Mac OS X
        WGET_MINICONDA="${WGET_MINICONDA}MacOSX-x86_64.sh"
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # GNU/Linux
        WGET_MINICONDA="${WGET_MINICONDA}Linux-x86_64.sh"
    else
        echo "Error: Unrecognized platform: $(uname -a)" 1>&2
        exit 1
    fi

    MINICONDA_SH="$(mktemp -d)/miniconda.sh"

    wget ${WGET_MINICONDA} -O ${MINICONDA_SH};

    bash ${MINICONDA_SH} -b -p $HOME/miniconda

    # Remove install script on exit.
    trap "rm -rf ${MINICONDA_SH}" EXIT
fi


# http://conda.pydata.org/docs/travis.html
export PATH="${MINICONDA}/bin:$PATH"
hash -r

cd ${PROJ_DIR}

conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

conda create -q -n dshinpy python=$PYTHON_VERSION || true

source activate dshinpy

conda install --file conda_requirements.txt

yes | pip install --upgrade -r requirements.txt

