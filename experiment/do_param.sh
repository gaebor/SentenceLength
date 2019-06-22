#!/bin/bash

filename=`cut -f1 -d" " <<<"$1"`
param="$2"

echo -n "$filename	$param	"

if [[ $param == o[1-3]\.k[1-9\.]* ]]
then
  args="--iter 1000 --max 1000 --eta 0.5 --opt Adagrad --mae 1e-3"
  order=`echo $param | sed "s/o\([1-9]\+\)\.k\([0-9\.]\+\)/\1/"`
  k="`echo $param | sed "s/o\([1-9]\+\)\.k\([0-9\.]\+\)/\2/" | tr '.' ' '`"
  echo `THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python randomwalk.py -o $order -k $k $args -f "$filename" &> >(tail -n 1)`
else
  if [[ $param == o[1-9]\.c[1-9\.]* ]]
  then
    echo couple
  else
    if [[ $param == b[1-9\.]* ]]
    then
      bins="`echo $param | sed "s/^b//" | tr '.' ' '`"
      echo `THEANO_FLAGS=device=cpu OMP_NUM_THREADS=1 python bins.py -b $bins --max 1000 -f "$filename" &> >(tail -n 1)`
    else
      if [[ $param == g-0.[1-9]* ]]
      then
        gamma="`echo $param | sed "s/^g//"`"
        #                  dataset   xmax  tol    eta iter usehessian gamma
        echo `./sichel.m "$filename" 1000 "10^-4" 0.4 100  True      "$gamma" &> >(tail -n 1)`
      else
        echo unknown model
      fi
    fi
  fi
fi
