#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [ "$1" == "-h" ]
then
  echo "Usage: `basename $0` [-h]"
  echo Deletes those files for which the corresponding line of stdin was not valid.
  echo Valid line looks something like this: "filename\\torder\\tk\\tnumbers"
  echo 
  echo "For example is you have \"file.stat	2	1.2.3	WARNING...\""
  echo "then \"file.stat.o2.k1.2.3.learned\" will be deleted."
  exit 0
fi

$DIR/badlines.sh | cut -f-3 -d "	" | \
while read -a line
do
    filename="${line[0]}.o${line[1]}.k${line[2]}.learned"
    echo $filename
    rm $filename
done
