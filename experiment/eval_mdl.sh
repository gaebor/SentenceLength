#!/bin/bash
filename=`cut -f1 -d" " <<<"$1"`
tolerance=`cut -f2 -d" " <<<"$1"`

echo -n "$filename	$2	$3	"

result="`THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python fit.py \
    --load $filename.o$2.k$3.learned --mdl $tolerance --iter 0 --max 1000 \
    < $filename 2> >((echo; cat) | tail -n 1)`"

exitstatus=$?

if [[ $exitstatus -eq 0 ]]
then
    echo "$result" | tail -n 2 | head -n 1 | sed 's/^[ \t]*//'
else
    echo "$result" | tail -n 1 | sed 's/^[ \t]*//'
    exit $exitstatus
fi
