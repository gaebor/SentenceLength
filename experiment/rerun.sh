#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

COMMAND="$DIR/do_param.sh"
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

$DIR/badlines.sh | cut -f-2 -d "	" | parallel -C "	" $PARGS "$COMMAND"
