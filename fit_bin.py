# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import argparse

import math
import numpy
import theano
import theano.tensor as T

from utils import read_stats, write_model, load_model, log_simplex_volume, logfactorial
from utils import constraint_mtx

def main(args):
    x_data, y_data = read_stats(sys.stdin, args.xmin, args.xmax,
                                normalize=True, swap=args.swap)
    y_data = theano.shared(y_data)
    x_min = numpy.min(x_data)
    x_max = numpy.max(x_data)
    
    bins = set(args.bins)
    if len(bins) != len(args.bins):
            print("duplicate in bins!", file=sys.stderr)
            return 1
    bins |= set([x_min, x_max+1])
    bins = sorted(bins)
    if bins[0] < x_min or bins[-1] > x_max + 1:
        print("bin is out of range [{}, {})".format(x_min, x_max), file=sys.stderr)
        return 1
    print(bins, file=sys.stderr)
    bins = numpy.array(bins)
    widths = bins[1:] - bins[:-1]
    
    x = T.vector()
    p = T.nnet.softmax(x.reshape((1, -1))).reshape((-1,))
    
    q = T.concatenate([T.zeros(bins[0])] + [T.ones(widths[i])*p[i]/float(widths[i]) for i in range(len(widths))])
    
    error = y_data.dot(T.log(y_data/q[x_data]))

    import thextensions
    
    optimizer = eval("thextensions." + args.opt + "Optimizer")(error, x, eta=args.eta)
    optimizer.init(numpy.zeros(len(widths)))
    
    f_getter = theano.function([], [p, q],
                        updates=optimizer.renormalize_updates(),
                        givens=optimizer.givens())
    
    if args.iter > 0:
        digits = math.ceil(math.log10(args.iter + 1))
        formatstr = "\t{:0{}d}\t{:.4e}\t{:.4e}"
        MAE = T.as_tensor_variable([abs(y).max() for y in optimizer.grads]).max()
        f_update = theano.function([], [error, MAE],
                        updates=optimizer.updates(),
                        givens=optimizer.givens())
        mae = float('inf')
        i = 1
        print("\titer\tobjective\tMAE", file=sys.stderr)
        while mae > args.mae and i <= args.iter:
            objective, mae = f_update()
            print(formatstr.format(i, digits, objective, mae), file=sys.stderr)
            i += 1
            if not numpy.isfinite(objective):
                break
        p_learned, generated_probs = f_getter()
        print(*p_learned)
        
    if args.eval:
        KL_term = error.eval({p: p_learned})
        common_entropy_term = 0.0
        model_volume = log_simplex_volume(len(widths))
        aux_model_volume = 0.0
        model_hessian = thextensions.Hessian(error, p).eval({p: p_learned})
        J = constraint_mtx(len(widths))
        lndethessian = numpy.linalg.slogdet(J.transpose().dot(model_hessian).dot(J))
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
Reads data from stdin: \"length frequency\\n\" format
(\"frequency length\\n\" if swap is on).
Writes the result to stdout and optimization info to stderr.
Author: Gábor Borbély, License: MIT
Contact: http://math.bme.hu/~borbely/indexeng""",
                formatter_class=MyFormatter)

    parser.add_argument('-m', "--max", dest='xmax', type=int, default=100,
                    help='maximum sentence length')
    
    parser.add_argument("-min", "--min", dest='xmin', type=int, default=1,
                    help='minimum sentence length')
                    
    parser.add_argument('-i', "--iter", dest='iter', type=int, default=100,
                    help="Maximum number of steps for the gradient descent")
    
    parser.add_argument('-mae', "--mae", dest='mae', type=float, default=1e-3,
                    help="The gradient descent stops if the Max Absolute Error of the gradient is below this threshold," + 
                         "if not positive then the MAE of the gradient is irrelevant.")
                    
    parser.add_argument('-e', "--eta", dest='eta', type=float, default=1,
                    help='learning rate')

    parser.add_argument("--opt", "--optimizer", dest='opt', type=str,
                    choices=["Adagrad", "GradientDescent"], default="Adagrad")                
    
    parser.add_argument('-eval', '--eval', "--evaluate", dest='eval', action="store_true",
                    default=False, help='evaluate the learned model')

    parser.add_argument('-s', '--swap', dest='swap', default=False,
                    help='swap columns in input', action='store_true')

    parser.add_argument('-b', '--bin', '--bins', dest='bins', default=[],
                    help='bins to use', type=int, nargs="*")
                
    exit(main(parser.parse_args()))
