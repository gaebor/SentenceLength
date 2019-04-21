#!/bin/bash

if [ "$1" == "-h" ]
then
  echo "Usage: `basename $0` [-h|-g|--good] [additional]"
  echo "Greps every bad line from stdin."
  echo "Optionally the good lines."
  echo 
  echo "\"additional\" specifies an additional string to accept (or reject in the default case)"
  
  exit 0
fi

GOOD="-v"

ADDITIONAL="0001"

if [[ "$1" == "-g" || "$1" == "--good" ]]
then
    GOOD=""
fi

if [ -n "$2" ]
then
    ADDITIONAL="$2"
fi

grep $GOOD -P "[^\t]+\t\d\t\d(\.\d)*\t([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?|$ADDITIONAL)"
