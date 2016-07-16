#!/usr/bin/env bash

PYTHON_VERSION=3.4
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

command_exists () {
    type "$1" &> /dev/null ;
}

LIBC_DIR="$HOME/libc_env/"


# EWS only. libc is outdated.
if [ ! -d ${LIBC_DIR} ]; then
if command_exists rpm2cpio ; then
    mkdir $LIBC_DIR
    cd $LIBC_DIR
    wget http://launchpadlibrarian.net/137699828/libc6_2.17-0ubuntu5_amd64.deb
    wget http://launchpadlibrarian.net/137699829/libc6-dev_2.17-0ubuntu5_amd64.deb
    wget ftp://rpmfind.net/linux/sourceforge/m/ma/magicspecs/apt/3.0/x86_64/RPMS.lib/libstdc++-4.8.2-7mgc30.x86_64.rpm
    ar p libc6_2.17-0ubuntu5_amd64.deb data.tar.gz | tar zx
    ar p libc6-dev_2.17-0ubuntu5_amd64.deb data.tar.gz | tar zx
    rpm2cpio libstdc++-4.8.2-7mgc30.x86_64.rpm| cpio -idmv
fi
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

conda install -q --file conda_requirements.txt

yes | pip install -r requirements.txt

