#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [[ -z "$1" || "$1" == "-h" ]]
then
    echo Bins model.
    echo Runs every file in the dataset through every possible parameter combinations.
    echo "USAGE: `basename $0` dataset binsfile"
    echo "You must provide a dataset file!"
    echo "Takes parallel ssh config from \"~/.parallel/sshloginfile\""
    exit
fi

parallel --eta --sshloginfile .. \
    "source work; nice bash $DIR/do_bins.sh 2>&1" \
    :::: "$1" \
    :::: "$2"
