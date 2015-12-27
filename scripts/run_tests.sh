#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -xe

python3 -m unittest discover -s $DIR/../dshin/
