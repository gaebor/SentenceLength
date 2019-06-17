#!/usr/bin/env bash
USE_TOL=0
USE_MDL=0

N_VALUES="1000 10000 100000 1000000 1000000000 inf"

while test $# -gt 0; do
    case "$1" in
        -t|--tolerance|--tol)
            export USE_TOL=1
            shift
            ;;
        -MDL|-mdl|--mdl|--MDL)
            export USE_MDL=1
            shift
            ;;
        -n|--n|-N|--N)
            export N_VALUES="$2"
            shift
            shift
            ;;
        *)
            break
            ;;
    esac
done

dataset="$1"
shift

echo $@

if [[ $USE_MDL -eq 0 ]]
then
    echo FILENAME $N_VALUES
fi
while read -r -a line
do
    echo -n "${line[0]}"
    tol=${line[1]}
    for n in `echo $N_VALUES`
    do
        if [[ $USE_MDL -gt 0 ]]
        then
            read -r -a result <<<"`grep -h ${line[0]} $@ | python interpret_results.py 2> /dev/null`"
        else
            if [[ $USE_TOL -gt 0 ]]
            then
                read -r -a result <<<"`grep -h ${line[0]} $@ | python interpret_results.py $n $tol`"
            else
                read -r -a result <<<"`grep -h ${line[0]} $@ | python interpret_results.py $n 0`"
            fi
        fi
        echo -n " ${result[1]}"
        # read -r -a best <<<"`grep "${line[0]}	${result[1]}	${result[2]}	" < $2 | cut -f4 -d "	" | cut -f1,2 -d " "`"
        # # echo -n $best
        # echo -n "(`python -c "print((${best[0]}+${best[1]})/$tol)"`)"
        # # `python -c "print(${best[3]}/$tol)"`
        if [[ $USE_MDL -gt 0 ]]
        then
            echo -n " ${result[3]} ${result[4]}"
            break
        fi
    done
    echo
done < "$dataset"
