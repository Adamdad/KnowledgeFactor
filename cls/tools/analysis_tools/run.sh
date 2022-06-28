#!/usr/bin/env bash

PYTHON_FILE=$1
CONFIG=$2


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/$PYTHON_FILE $CONFIG