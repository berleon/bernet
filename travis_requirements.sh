#! /usr/bin/env bash

ANACONDA_PACKAGES=$(./anaconda_packages.sh)
function join { local IFS="$1"; shift; echo "$*"; }

REGEXP=$(printf '|%s==' ${ANACONDA_PACKAGES[@]})
REGEXP=${REGEXP:1}

egrep -v "$REGEXP" requirements.txt