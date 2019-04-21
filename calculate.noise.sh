for file in $1
do
    directory=`dirname $file`
    filename=`basename $file $2`
    for file1 in $directory/$filename*$2
    do
        echo -n $file1
        for file2 in $directory/$filename*$2
        do
            echo -n "	`python gKL.py -P -Q $file1 $file2 2> /dev/null`"
        done
        echo
    done
    echo
done
