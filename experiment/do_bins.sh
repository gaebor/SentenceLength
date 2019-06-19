#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

filename=`cut -f1 -d" " <<<"$1"`

echo -n "$filename	b$2	"

bins="`tr "." " " <<<$2`"

THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python bins.py \
        -b $bins --max 1000 -f "$filename" &> >(tail -n 1)
