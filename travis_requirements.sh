#! /usr/bin/env bash

ANACONDA_PACKAGES=$(./anaconda_packages.sh)
function join { local IFS="$1"; shift; echo "$*"; }

REGEXP=$(join '|' ${ANACONDA_PACKAGES[@]})
egrep -v "$REGEXP" requirements.txt