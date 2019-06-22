# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import argparse

import math
import numpy

from utils import *

def save_model(k, alpha, p, x, probs, file=sys.stdout):
    order = p.shape[1] # number of steps upward + 1
    steps = numpy.arange(-1, order-1)
    print("# k alpha m " + " ".join(("p%d" % i) for i in steps), file=file)
    for i in range(len(k)):
        print("#", k[i], alpha[i], p[i].dot(steps), *p[i], file=file)
    print("# average values", alpha.dot(p.dot(steps)), *alpha.dot(p), file=file)

    for i in range(len(x)):
        print(probs[i], x[i], file=file)

def load_model(file_name):
    with open(file_name, "r") as file:        
        comment_reader = (line[1:].strip().split() for line in file if line[0] == "#")
        
        header = next(comment_reader)
        if len(header) > 4:
            # k alpha m pm1 p0 ...
            
            k = []
            alpha = []
            p = []
            
            for line in comment_reader:
                try:
                    k.append(int(line[0]))
                    alpha.append(float(line[1]))
                    p.append([float(x) for x in line[3:]])
                except:
                    continue
        else:
            raise ValueError("not a model")
        
        return (numpy.array(k, dtype='int32'),
                numpy.array(alpha, dtype=tconfig.floatX),
                numpy.array(p, dtype=tconfig.floatX))

def main(args):
    learned = args.filename + ".o" + str(args.order) + ".k" + ".".join(map(str, args.k)) + ".learned"
    try:
        k, alpha_initial, x_initial = load_model(learned)
        alpha_initial = numpy.log(alpha_initial)
        x_initial = numpy.log(x_initial)
        if (k != numpy.array(args.k, dtype='int32')).any() or \
            x_initial.shape[1] != args.order + 2:
            print("Model \"" + learned + "\" has order", x_initial.shape[1]-2, "and k", k, file=sys.stderr)
            raise
        order = args.order + 2
    except:
        order = args.order + 2 # span of steps
        k = numpy.sort(numpy.abs(numpy.array(args.k, dtype="int32")))
        if args.random:
            x_initial = numpy.random.rand(1 if args.coupled else len(k), order).astype(tconfig.floatX)
            alpha_initial = numpy.random.rand(len(k)).astype(tconfig.floatX)
        else:
            x_initial = numpy.zeros((1 if args.coupled else len(k), order)).astype(tconfig.floatX)
            alpha_initial = numpy.zeros(len(k)).astype(tconfig.floatX)
    
    k_min = min(k)
    
    if k_min <= 0:
        print("ERROR: k values should be positive integers!", file=sys.stderr)
        return 1

    x_data, y_data = read_stats(args.filename, args.xmin, args.xmax,
                                normalize=True, swap=args.swap)
    x_min = numpy.min(x_data)
    x_max = numpy.max(x_data)

    # cut off
    intersection = [i for i in range(len(x_data)) if x_data[i] >= k_min]
    left_out = [i for i in range(len(x_data)) if x_data[i] < k_min]
    
    x_common = x_data[intersection]
    y_common = y_data[intersection]
    
    entropy_nat = -y_data.dot(numpy.log(y_data))
    
    covered_weight = y_common.sum()
    left_out_dim = len(x_data) - len(intersection)

    import theano
    import theano.tensor as T
    
    N = x_max * (order - 1) + 1 # size of the representation matrix
    x = T.matrix()
    p = T.nnet.softmax(x)
    
    #                n
    #             [X    ]
    #             [X X  ]
    #          2n [X X X]
    #             [  X X]
    #             [    X]
    #      2n     [     ]
    # [Y          ]
    # [Y Y        ]
    # [Y Y Y      ]
    # [Y Y Y Y    ]
    # [  Y Y Y Y  ]
    # [    Y Y Y Y]
    # [      Y Y Y]
    # [        Y Y]
    # [          Y]
    #  .
    #  .
    #  .

    n = order
    # E: n * 2n * n
    E = T.stack([T.eye(2*n, n, k=-i) for i in range(n)], axis=0)

    # E0: len(k) * N * n
    E0 = T.ones(len(k))[:, None, None]*T.eye(N, n)[None, :, :]

    # P: len(k) * 2n * n
    P = (p[:, :, None, None]*E[None, :, :, :]).sum(1)

    def duplicate(x):
        '''
        makes a k*N*2n array from a k*N*n array
        [X  ]           [X  |0 0]
        [X X]  ->       [X X|0 0]
        [  X]           [  X|X  ]
        [   ]           [   |X X]
        '''
        x_ = x[:, :-n, :]
        x__ = T.concatenate([T.zeros((len(k), n, n)), x_], axis=1)
        return T.concatenate([x, x__], axis=2)

    # x_max * len(k) * N * n
    powers, updates1 = theano.scan(
                            lambda x, P_: T.batched_dot(duplicate(x), P_),
                            n_steps=x_max, outputs_info=E0,
                            strict=True, non_sequences=[P])

    # x_max * len(k) * N
    modelled_probs = powers[:, :, :, 0]
    
    # x_common is a list of indices
    probs = modelled_probs[x_common - 1, :, :]

    i_ = [(i*len(k) + j)*N + x_common[i] - k[j] if x_common[i] >= k[j] else -1 for i in range(len(x_common)) for j in range(len(k))]

    # len(x_common) * len(k)
    probs_ = T.concatenate([probs.reshape((-1,)), [0.0]])
    probs = probs_[numpy.array(i_, dtype='int32')].reshape((len(x_common), len(k)))
    
    x_shared = theano.shared(x_common)
    # k/n coefficient
    probs = probs*(theano.shared(k.astype(tconfig.floatX))[None, :]/x_shared[:, None])

    alpha_ = T.vector()
    alpha = T.nnet.softmax(alpha_.reshape((1, -1))).reshape((-1,))
    generated_probs = probs.dot(alpha)

    y_shared = theano.shared(y_common)
    
    error = y_shared.dot(T.log(y_shared/generated_probs))

    import thextensions
    
    optimizer = eval("thextensions." + args.opt + "Optimizer")(error, x, alpha_, eta=args.eta)
    optimizer.init(x_initial, alpha_initial)
    
    if args.iter > 0:
        digits = math.ceil(math.log10(args.iter + 1))
        formatstr = "\t{:0{}d}\t{:.4e}\t{:.4e}"
        
        MAE = T.as_tensor_variable([abs(y).max() for y in optimizer.grads]).max()
        f_update = theano.function([], [error, MAE],
                        updates=updates1 + optimizer.updates(),
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
        f_getter = theano.function([], [alpha, p, generated_probs],
                        updates=updates1 + optimizer.renormalize_updates(),
                        givens=optimizer.givens())
        alpha_learned, p_learned, generated_probs = f_getter()

        with open(learned, "w") as outfile:
            save_model(k, alpha_learned, p_learned, x_common, generated_probs, file=outfile)

    # evaluate
    k = x_initial.shape[0]
    # number of parameters
    d = k * (order - 1) + k - 1
    if left_out_dim > 0:
        d += left_out_dim - 1
        common_entropy_term = -covered_weight*numpy.log(covered_weight) if covered_weight > 0 else 0.0
    else:
        common_entropy_term = 0.0

    f_eval = theano.function([], error,
                    updates=updates1,
                    givens=optimizer.givens())
    
    f_hessian = theano.function([], thextensions.Hessian(error, p, alpha),
                    updates=updates1,
                    givens=optimizer.givens())

    objective = f_eval()
    
    left_out_hessian = 0.0
    left_out_volume = 0.0
    
    if left_out_dim > 0:
        left_out_volume = log_simplex_volume(left_out_dim)
        if left_out_dim > 1:
            # diagonal
            left_out_hessian = (1.0-covered_weight)**2/y_data[left_out]
            J_aux = constraint_mtx(left_out_dim)
            left_out_hessian = logdet(J_aux.transpose().dot(left_out_hessian[:, None]*J_aux))
    
    model_volume = k * log_simplex_volume(order) + log_simplex_volume(k)
    
    J = numpy.zeros((k*order + k, k*order-1), dtype=tconfig.floatX)
    for i in range(k):
        J[i*order:(i+1)*order, i*(order-1):(i+1)*(order-1)] = constraint_mtx(order)
    if k > 1:
        J[-k:, -(k-1):] = constraint_mtx(k)

    hessian = logdet(J.transpose().dot(f_hessian()).dot(J))
    print(objective, common_entropy_term, model_volume, left_out_volume,
              hessian, left_out_hessian, d, file=sys.stdout)

    return 0

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
                description="""                                    .
                                    .
                                    p2
    ^                             / p1
    |            the steps can be - p0
    |  _    /\                    \ p-1
k=3 |_/ \  /  \                   
    |    \/    \_         
    |            \        
  0 +-------------+----> 
              return time 

Fits mixture of Markovian model's return times to data.
Author: Gábor Borbély, License: MIT
Contact: http://math.bme.hu/~borbely/indexeng

Reads data from a given file: \"length frequency\\n\" format
(\"frequency length\\n\" if swap is on).
Writes the result to "filename.o9.k9.learned" and optimization info to stderr.
The one and only line of the stdout should be evaluation info.""",
                formatter_class=MyFormatter,
                epilog="""During the evaluation the outputs are the followings (in order):
    {0} KL
    {1} common entropy term
    {2} log model volume
    {3} log volume of auxiliary model
    {4} log det Hessian
    {5} log det Hessian of auxiliary model
    {6} number of model parameters
    
The total cost is the following, for a given data size n:

                 {2} + {3}     {4} + {5}     {6}         n
    {0} + {1} + ----------- + ----------- + ----- * log ----
                     n           2 * n      2 * n       2*pi
""")

    parser.add_argument('-m', "--max", dest='xmax', type=int, default=100,
                    help='maximum sentence length')
    
    parser.add_argument("-min", "--min", dest='xmin', type=int, default=1,
                    help='minimum sentence length')
                    
    parser.add_argument('-k', dest='k', type=int, nargs="+", default=[3],
                    help='k values in the mixture')
    
    parser.add_argument('-o', "--order", dest='order', type=int, default=2,
                    help='how many steps can be taken upwards')
    
    parser.add_argument('-i', "--iter", dest='iter', type=int, default=10,
                    help="Maximum number of steps for the gradient descent")
    
    parser.add_argument('-mae', "--mae", dest='mae', type=float, default=1e-3,
                    help="The gradient descent stops if the Max Absolute Error of the gradient is below this threshold," + 
                         "if not positive then the MAE of the gradient is irrelevant.")

    parser.add_argument('-e', "--eta", dest='eta', type=float, default=1,
                    help='learning rate')

    parser.add_argument("--opt", "--optimizer", dest='opt', type=str,
                    choices=["Adagrad", "GradientDescent", "Hessian"], default="Adagrad")                
    
    parser.add_argument('-f', "--filename", dest='filename', type=str, default="",
                    help='data filename, model filename is inferred')
    
    parser.add_argument('-r', "--random", dest='random', default=False,
                    help='random initial model (otherwise uniform)', action='store_true')

    parser.add_argument('-c', "--coupled", dest='coupled', default=False,
                    help='couple probabilities of different k values'+
                        '(if more than one k value is used)', action='store_true')

    parser.add_argument('-s', '--swap', dest='swap', default=False,
                    help='swap columns in input', action='store_true')

    exit(main(parser.parse_args()))
