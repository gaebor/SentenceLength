#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

COMMAND="$DIR/do_param.sh"

if [[ -z "$1" || "$1" == "-h" ]]
then
    echo Runs every file in the dataset through every argument.
    echo "USAGE: `basename $0` [-c command] dataset args [args ...]"
    echo "Default command is \"$COMMAND\""
    echo "Takes parallel ssh config from \"~/.parallel/sshloginfile\""
    exit
fi

if [[ "$1" == "-c" || "$1" == "--command" ]]
then
    COMMAND="$2"
    shift
    shift
fi

dataset="$1"
shift

parallel --eta --sshloginfile .. \
    "source work; nice $COMMAND" \
    :::: "$dataset" \
    :::: $@
