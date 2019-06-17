#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

filename=`cut -f1 -d" " <<<"$1"`
# tolerance=`cut -f2 -d" " <<<"$1"`
learned="$filename.o$2.k$3.learned"

echo -n "$filename	o$2.k$3	"

bash "$DIR/save_result.sh" "/dev/null" THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python fit.py \
        --load \"$filename.o$2.k$3.learned\" --eval --iter 0 --max 1000 < "$filename"

exit $?
