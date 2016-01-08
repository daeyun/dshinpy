#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -xe

cd $DIR/../

export PYTHONPATH="$HOME/anaconda/lib/python3.5/site-packages/:$PYTHONPATH"

python3 -OO -m unittest discover -s $DIR/../dshin/
