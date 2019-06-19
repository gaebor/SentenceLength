# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import argparse

import math
import numpy

from utils import *

def main(args):
    bins = sorted(args.bins)
    if len(set(bins)) != len(bins):
            print("duplicate in bins!", file=sys.stderr)
            return 1

    x_data, y_data = read_stats(args.filename, args.xmin, args.xmax,
                                normalize=True, swap=args.swap)
    x_min = numpy.min(x_data)
    x_max = numpy.max(x_data)
    
    bins = [x_min] + bins + [x_max+1]
    
    bins = numpy.array(bins)
    widths = numpy.maximum(bins[1:] - bins[:-1], 0)
    
    bin_probs = numpy.array([y_data[(bins[i] <= x_data) & (x_data < bins[i+1])].sum() for i in range(len(bins)-1)])
    
    q = numpy.concatenate([numpy.zeros(max(bins.min(), 0))] + [numpy.ones(widths[i])*bin_probs[i]/float(widths[i]) for i in range(len(widths))])
    
    KL_term = y_data.dot(numpy.log(y_data/q[x_data]))
    common_entropy_term = 0.0
    model_volume = log_simplex_volume(len(widths))
    aux_model_volume = 0.0
    # diagonal
    model_hessian = numpy.array([0 if x == 0 else 1.0/x for x in bin_probs])
    J = constraint_mtx(len(widths))
    lndethessian = numpy.linalg.slogdet(J.transpose().dot(model_hessian[:, None]*J))
    if lndethessian[0] > 0:
        lndethessian = lndethessian[1]
    else:
        lndethessian = float("inf")
    aux_model_hessian = 0.0
    number_of_parameters = len(widths)-1
    
    print(KL_term, common_entropy_term, model_volume, aux_model_volume,
            lndethessian, aux_model_hessian, number_of_parameters, file=sys.stderr)
    return 0

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
                description="""Fits bins to a given distribution.
Reads data from 'filename' in a \"length frequency\\n\" format
(\"frequency length\\n\" if 'swap' is on).
Writes Bayesian model comparison info to stdout.

Author: Gábor Borbély, License: MIT
Contact: http://math.bme.hu/~borbely/indexeng""",
                formatter_class=MyFormatter)

    parser.add_argument('-m', "--max", dest='xmax', type=int, default=100,
                    help='maximum sentence length')
    
    parser.add_argument("-min", "--min", dest='xmin', type=int, default=1,
                    help='minimum sentence length')
    
    parser.add_argument('-s', '--swap', dest='swap', default=False,
                    help='swap columns in input', action='store_true')

    parser.add_argument('-f', "--filename", dest='filename', type=str, default="",
                    help='data filename, model filename is inferred')
    
    parser.add_argument('-b', '--bin', '--bins', dest='bins', default=[],
                    help='bins to use', type=int, nargs="*")
                
    exit(main(parser.parse_args()))
