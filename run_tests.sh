#! /usr/bin/env bash

nosetests \
    --with-timer \
    --timer-top-n 10 \
    --with-coverage \
    --cover-package=bernet \
    --cover-html-dir=htmlcov \
    --cover-html \
    --cover-min-percentage=90
