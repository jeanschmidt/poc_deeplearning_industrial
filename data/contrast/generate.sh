#!/bin/bash

CNV="convert"
OPTS=" -normalize -gamma 2 -auto-level -background "#000" -rotate 4 -crop 622x234+15+89 "

COMP="composite"
COMP_OPTS=" -compose minus "
COMP_BASE="../filterout/filterout.jpg"
TEMP_FILE="temp.jpg"

SRC="../original"
GOOD="good"
BAD="bad"

if [ ! -d ${GOOD} ] ; then
    if [ -e ${GOOD} ] ; then
        rm -rf ${GOOD}
    fi
    mkdir -p ${GOOD}
fi

echo -n "GOOD"
I=0
for f in ${SRC}/${GOOD}/* ; do
    ${COMP} $f "${COMP_BASE}" ${COMP_OPTS} "${TEMP_FILE}"
    ${CNV} "${TEMP_FILE}" ${OPTS} ${GOOD}/${I}.jpg
    rm -f "${TEMP_FILE}"
    #${CNV} $f ${OPTS} ${GOOD}/${I}.jpg
    I=$(( $I + 1 ))
    echo -n "."
done
echo " OK!"

if [ ! -d ${BAD} ] ; then
    if [ -e ${BAD} ] ; then
        rm -rf ${BAD}
    fi
    mkdir -p ${BAD}
fi

echo -n "BAD"
I=0
for f in ${SRC}/${BAD}/* ; do
    ${COMP} $f "${COMP_BASE}" ${COMP_OPTS} "${TEMP_FILE}"
    ${CNV} "${TEMP_FILE}" ${OPTS} ${BAD}/${I}.jpg
    rm -f "${TEMP_FILE}"
    #${CNV} $f ${OPTS} ${BAD}/${I}.jpg
    I=$(( $I + 1 ))
    echo -n "."
done
echo " OK!"


