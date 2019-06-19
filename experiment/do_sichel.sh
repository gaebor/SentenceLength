#!/bin/bash

filename=`cut -f1 -d" " <<<"$1"`

echo -n "$filename	g$2	"

./sichel.m "$filename" 1000 "10^-4" 0.4 100 True "$2" &> >(tail -n 1)
