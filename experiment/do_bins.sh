#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

filename=`cut -f1 -d" " <<<"$1"`
learned="$filename.bins$2.learned"

args="--iter 1000 --max 1000 --eta 1 --opt Adagrad --mae 5e-4"

echo -n "$filename	b$2	"

bins="`tr "." " " <<<$2`"

bash "$DIR/save_result.sh" "$learned" THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python fit_bin.py \
        -eval -b $bins $args < "$filename"

exit $?
