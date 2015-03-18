#! /usr/bin/env bash

set -e

./run_tests.sh --without-examples
(cd docs;
make html)


