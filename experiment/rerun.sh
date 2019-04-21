#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

COMMAND=".experiment/learn_one.sh"
PARGS="-j 4 --eta"

if [ "$1" == "-h" ]
then
  echo "Usage: `basename $0` [-h|command] [args]"
  echo "Greps every bad line from stdin and runs \"command\" over it again."
  echo "Bad line means that it doesn't match anything like this: filename\\torder\\tk\\tnumbers"
  echo Default command is "\"$COMMAND\""
  echo Uses the second argument as additional arguments to parallel, default is "\"$PARGS\""

  exit 0
fi

if [ -n "$1" ]
then
    COMMAND="$1"
fi

if [ -n "$2" ]
then
    PARGS="$2"
fi

$DIR/badlines.sh | cut -f-3 -d "	" | \
while read -a line
    do echo "`grep ${line[0]} experiment/datasets.txt`	${line[1]}	${line[2]}"
done | parallel -C "	" $PARGS "$COMMAND"
