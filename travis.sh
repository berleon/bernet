#! /usr/bin/env bash

set -e

./run_tests.sh
(cd docs;
make html)


