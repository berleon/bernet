#! /usr/bin/env bash

nosetests --with-coverage \
    --cover-package=bernet \
    --cover-html-dir=htmlcov \
    --cover-html \
    --cover-min-percentage=90
