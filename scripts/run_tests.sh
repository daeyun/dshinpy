#!/bin/bash

# http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

set -xe

cd $DIR/../


MINICONDA="$HOME/miniconda"
export PATH="${MINICONDA}/bin:$PATH"
hash -r

source activate dshinpy

LIBC_DIR="$HOME/libc_env/"

if [ -d ${LIBC_DIR} ]; then

LD_LIBRARY_PATH="$LIBC_DIR/lib/x86_64-linux-gnu/:$LIBC_DIR/usr/lib64/:$LD_LIBRARY_PATH" \
  $LIBC_DIR/lib/x86_64-linux-gnu/ld-2.17.so $(which python) $(which py.test)

else

py.test

fi
