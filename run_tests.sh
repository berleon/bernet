#! /usr/bin/env bash

NOSE_OPT="
    --with-timer
    --timer-top-n 10
    --with-coverage
    --cover-erase
    --cover-package=bernet
    --cover-html-dir=htmlcov
    --cover-html
    --cover-min-percentage=90"

if [ "$1" == "--without-examples" ];
then
    nosetests --attr '!example' $NOSE_OPT
else
    nosetests $NOSE_OPT
fi

