#!/bin/bash
filename=`cut -f1 -d" " <<<"$1"`
learned="$filename.o$2.k$3.learned"

args="--iter 10000 --max 1000 --eta 0.1 --opt Adagrad --mae 1e-3"

echo -n "$filename	$2	$3	"

k="`tr "." " " <<<$3`"

if [[ -f $learned ]]
then
    result="`THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python fit.py \
        --load $learned $args < $filename 2> >(tail -n 1)`"
else
    result="`THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python fit.py \
        -o $2 -k $k $args < $filename 2> >(tail -n 1)`"
fi

exitstatus=$?

echo "$result" | tail -n 1 | sed 's/^[ \t]*//'

if [[ $exitstatus -eq 0 ]]
then
    echo "$result" | head -n-1 > $learned
else
    exit $exitstatus
fi
