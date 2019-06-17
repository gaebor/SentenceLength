# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import argparse

import math
import numpy

from fit import read_stats

def main(args):
    x_data, y_data = read_stats(sys.stdin, args.xmin, args.xmax,
                                normalize=False, swap=args.swap)
    x_min = numpy.min(x_data)
    x_max = numpy.max(x_data)
    
    total = y_data.sum()
    s = 0.0
    j = 1
    for i in range(len(x_data)):
        p = y_data[i]
        x = x_data[i]
        s += p
        
        if s*args.n > j*total:
            j += 1
            print(x, end=' ')
    
    print("")
    return 0

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
                description="""Finds equiprobabilistic partitions.
Author: Gábor Borbély, License: MIT
Contact: http://math.bme.hu/~borbely/indexeng""",
                formatter_class=MyFormatter)

    parser.add_argument('-m', "--max", dest='xmax', type=int, default=100,
                    help='maximum sentence length')
    
    parser.add_argument("-min", "--min", dest='xmin', type=int, default=1,
                    help='minimum sentence length')
               
    parser.add_argument('-s', '--swap', dest='swap', default=False,
                    help='swap columns in input', action='store_true')

    parser.add_argument('-n', dest='n', default=10,
                    help='number of bins', type=int)
                
    exit(main(parser.parse_args()))
