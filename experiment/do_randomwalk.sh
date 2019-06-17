#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

filename=`cut -f1 -d" " <<<"$1"`
learned="$filename.o$2.k$3.learned"

args="--iter 10000 --max 1000 --eta 0.5 --opt Adagrad --mae 1e-3 -eval"

echo -n "$filename	o$2.k$3	"

k="`tr "." " " <<<$3`"

if [[ -f $learned ]]
then
    bash "$DIR/save_result.sh" "$learned" THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python fit.py \
        --load \"$learned\" $args < "$filename"
else
    bash "$DIR/save_result.sh" "$learned" THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python fit.py \
        -o $2 -k $k $args < "$filename"
fi

exit $?
