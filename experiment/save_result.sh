#!/bin/bash

filename="$1"
shift

result="`eval $@ 2> >(tail -n 1)`"
exitstatus=$?

echo "$result" | tail -n 1 | sed 's/^[ \t]*//'

if [[ $exitstatus -eq 0 ]]
then
    echo "$result" | head -n-1 > "$filename"
else
    exit $exitstatus
fi
