#!/usr/bin/env bash

RIFFLE=0

distribution () {
    awk '{print NF}' | sort -n | uniq -c | sed "s/^ *\([0-9]\+\) \([^\n]*\)/\2\t\1/"
}

while test $# -gt 0; do
    case "$1" in
        -r|--riffle)
            export RIFFLE=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

for file in $@
do
    echo -n "$file.lengths "
    distribution < "$file" > "$file.lengths"
    if [ $RIFFLE -eq 1 ]
    then
        firstfile="$file.oddhalf.lengths"
        secondfile="$file.evenhalf.lengths"
        awk 'NR % 2 == 1' "$file" | distribution > "$firstfile"
        awk 'NR % 2 == 0' "$file" | distribution > "$secondfile"
    else
        firstfile="$file.firsthalf.lengths"
        secondfile="$file.secondhalf.lengths"
        LENGTH=`wc -l < "$file"`
        head -n $((LENGTH/2))       "$file" | distribution > "$firstfile"
        tail -n+$(((LENGTH/2) + 1)) "$file" | distribution > "$secondfile"
    fi
    
    gkl1=`python gKL.py -P -Q "$firstfile" "$secondfile" 2> /dev/null`
    result1=$?
    gkl2=`python gKL.py -P -Q "$secondfile" "$firstfile" 2> /dev/null`
    result2=$?
        
    if [ $result1 = "0" -a $result2 = "0" ]
    then
        echo "0.5*($gkl1+$gkl2)" | bc
    else
        echo inf
    fi
done
