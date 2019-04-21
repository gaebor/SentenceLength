#!/bin/bash

COMMAND=".experiment/learn_one.sh"

if [ -n "$2" ]
then
    COMMAND="$2"
fi

if [[ -z "$1" || "$1" == "-h" ]]
then
    echo Runs every file in the dataset through every possible parameter combinations.
    echo "USAGE: `basename $0` dataset [command]"
    echo "You must provide a dataset file!"
    echo "Default command is \"$COMMAND\""
    echo "Takes parallel ssh config from \"~/.parallel/sshloginfile\""
    exit
fi

parallel --eta --sshloginfile .. \
    "source work; nice $COMMAND 2>&1" \
    :::: "$1" \
    ::: 1 2 3 \
    ::: "1" "2" "3" "4" "5" "1.2" "1.3" "1.4" "1.5" "2.3" "2.4" "2.5" "3.4" "3.5" "4.5" "1.2.3" "1.2.4" "1.2.5" "1.3.4" "1.3.5" "1.4.5" "2.3.4" "2.3.5" "2.4.5" "3.4.5" "1.2.3.4" "1.2.3.5" "1.2.4.5" "1.3.4.5" "2.3.4.5" "1.2.3.4.5"
