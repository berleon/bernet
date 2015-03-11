#! /usr/bin/env bash

set -e

./run_tests.sh --fast
(cd docs;
make html)


