# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import argparse

import math
import numpy
import theano
import theano.tensor as T

def read_stats(f, xmin=1, xmax=100, normalize=False, swap=False):
    if type(f) == type(""):
        f = open(f, "r")
    if swap:
        data = [line.strip().split()[::-1] for line in f if line[0] != "#"]
    else:
        data = [line.strip().split() for line in f if line[0] != "#"]
    data = list(filter(lambda d: int(d[0])>=xmin and int(d[0])<=xmax, data))
    data.sort(key=lambda d: int(d[0]))
    data = numpy.array(data)
    
    x_data = data[:, 0].astype("int32")
    y_data = data[:, 1].astype(theano.config.floatX)
    
    factor = 1.0/y_data.sum() if normalize else 1.0
    return x_data, y_data * factor

def write_model(k, alpha, p, x, probs, file=sys.stdout):
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
                numpy.array(alpha, dtype=theano.config.floatX),
                numpy.array(p, dtype=theano.config.floatX))

def log_simplex_volume(d):
    return 0.5*numpy.log(d) - logfactorial(d-1)
    # return numpy.log(numpy.sqrt(d)/numpy.math.factorial(d-1))
def logfactorial(n):
    result = 0.0
    while n > 1:
        result += numpy.log(n)
        n-=1
    return result

def constraint_mtx(k):
    J = numpy.zeros((k, k-1), dtype=theano.config.floatX)
    for n in range(1, k):
        J[:n,n-1] = -1.0/numpy.sqrt(n*(n+1.0))
        J[n,n-1] = numpy.sqrt(n/(n+1.0))
    return J

def discretize(x, min_val, dx):
    return numpy.round((x - min_val)*dx)/dx + min_val

def main(args):
    if len(args.load) > 0:
        try:
            k, alpha_initial, x_initial = load_model(args.load)
            alpha_initial = numpy.log(alpha_initial)
            x_initial = numpy.log(x_initial)
            order = x_initial.shape[1]
        except:
            print("Cannot load", "\"" + args.load + "\"!", file=sys.stderr)
            return 1
    else:
        order = args.order + 2 # span of steps
        k = numpy.sort(numpy.abs(numpy.array(args.k, dtype="int32")))
        if args.random:
            x_initial = numpy.random.rand(len(k), order).astype(theano.config.floatX)
            alpha_initial = numpy.random.rand(len(k)).astype(theano.config.floatX)
        else:
            x_initial = numpy.zeros((len(k), order)).astype(theano.config.floatX)
            alpha_initial = numpy.zeros(len(k)).astype(theano.config.floatX)
    
    k_min = min(k)
    
    if k_min <= 0:
        print("ERROR: k values should be positive integers!", file=sys.stderr)
        return 1

    x_data, y_data = read_stats(sys.stdin, args.xmin, args.xmax,
                                normalize=False, swap=args.swap)
    x_min = numpy.min(x_data)
    x_max = numpy.max(x_data)

    number_of_datapoints = y_data.sum()
    y_data /= number_of_datapoints
    
    # cut off
    intersection = [i for i in range(len(x_data)) if x_data[i] >= k_min]
    left_out = [i for i in range(len(x_data)) if x_data[i] < k_min]
    
    x_common = x_data[intersection]
    y_common = y_data[intersection]
    
    entropy_nat = -y_data.dot(numpy.log(y_data))
    
    covered_weight = y_common.sum()
    left_out_dim = len(x_data) - len(intersection)
    
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
    probs = probs*(theano.shared(k.astype(theano.config.floatX))[None, :]/x_shared[:, None])

    alpha_ = T.vector()
    alpha = T.nnet.softmax(alpha_.reshape((1, -1))).reshape((-1,))
    generated_probs = probs.dot(alpha)

    y_shared = theano.shared(y_common)
    
    error = y_shared.dot(T.log(y_shared/generated_probs))

    import thextensions
    
    optimizer = eval("thextensions." + args.opt + "Optimizer")(error, x, alpha_, eta=args.eta)
    optimizer.init(x_initial, alpha_initial)
    
    f_getter = theano.function([], [alpha, p, generated_probs],
                        updates=updates1 + optimizer.renormalize_updates(),
                        givens=optimizer.givens())
    
    if args.iter > 0:
        digits = math.ceil(math.log10(args.iter + 1))
        formatstr = "\t%%0%dd\t%%.4e" % digits
        if args.mae <= 0:
            f_update = theano.function([], error,
                            updates=updates1 + optimizer.updates(),
                            givens=optimizer.givens())
            print("\titer\tobjective", file=sys.stderr)
            for i in range(int(args.iter)):
                objective = f_update()
                print(formatstr % (i + 1, objective), file=sys.stderr)
                if not numpy.isfinite(objective):
                    break
        else:
            formatstr += "\t%.4e"
            MAE = T.as_tensor_variable([abs(y).max() for y in optimizer.grads]).max()
            f_update = theano.function([], [error, MAE],
                            updates=updates1 + optimizer.updates(),
                            givens=optimizer.givens())
            mae = float('inf')
            i = 1
            print("\titer\tobjective\tMAE", file=sys.stderr)
            while mae > args.mae and i <= args.iter:
                objective, mae = f_update()
                print(formatstr % (i, objective, mae), file=sys.stderr)
                i += 1
                if not numpy.isfinite(objective):
                    break
        alpha_learned, p_learned, generated_probs = f_getter()

        write_model(k, alpha_learned, p_learned, x_common, generated_probs, file=sys.stdout)

    # number of parameters
    d = len(k) * (order - 1) + len(k) - 1
    if left_out_dim > 0:
        d += left_out_dim - 1
        common_entropy_term = -covered_weight*numpy.log(covered_weight) if covered_weight > 0 else 0.0
    else:
        common_entropy_term = 0.0

    if args.eval:
        f_eval = theano.function([], error,
                        updates=updates1,
                        givens=optimizer.givens())
        
        f_hessian = theano.function([], thextensions.Hessian(error, p, alpha),
                        updates=updates1,
                        givens=optimizer.givens())
        
        objective = f_eval()
        
        if left_out_dim > 0:
            left_out_hessian = -numpy.log(y_data[left_out]).sum()
            left_out_volume = log_simplex_volume(left_out_dim)
        else:
            left_out_hessian = 0.0
            left_out_volume = 0.0

        model_volume = len(k) * log_simplex_volume(order) + log_simplex_volume(len(k))
        
        J = numpy.zeros((len(k)*order + len(k), len(k)*order-1), dtype=theano.config.floatX)
        for i in range(len(k)):
            J[i*order:(i+1)*order, i*(order-1):(i+1)*(order-1)] = constraint_mtx(order)
        if len(k) > 1:
            J[-len(k):, -(len(k)-1):] = constraint_mtx(len(k))

        hessian = numpy.linalg.slogdet(J.transpose().dot(f_hessian()).dot(J))
        if hessian[0] > 0:
            hessian = hessian[1]
        else:
            hessian = float("inf")
        
        print(objective, common_entropy_term, model_volume, left_out_volume,
                  hessian, left_out_hessian, d, file=sys.stderr)
    if args.mdl > 0:
        f_eval = theano.function([], error,
                        updates=updates1,
                        givens=optimizer.givens())
        
        tolerance = args.mdl

        alpha_learned, p_learned, _ = f_getter()
                    
        x_learned = numpy.log(p_learned)
        log_alpha_learned = numpy.log(alpha_learned)

        left_out_probs = y_data[left_out]
        left_out_log_probs = numpy.log(left_out_probs)
        
        log_min = x_learned.min()
        log_max = x_learned.max()
        
        if len(k) > 1: # more than one mixture
            log_min = min(log_min, log_alpha_learned.min())
            log_max = max(log_max, log_alpha_learned.max())
        if left_out_dim > 1: # more than one uncovered probability
            log_min = min(log_min, left_out_log_probs.min())
            log_max = max(log_max, left_out_log_probs.max())
                 
        bits_min = float("inf")
        for b in range(1, 33):
            # do the discretization
            dx = float(2**b-1)/(log_max - log_min)
            x_disc = discretize(x_learned, log_min, dx)
            alpha_disc = discretize(log_alpha_learned, log_min, dx)
            left_out_log_disc = discretize(left_out_log_probs, log_min, dx)
            
            left_out_log_disc += numpy.log(left_out_probs.sum()) - \
                                 numpy.log(numpy.exp(left_out_log_disc).sum())
            
            optimizer.init(x_disc, alpha_disc)
            
            # discretization effects auxiliary model too
            discretized_aux_kl = left_out_probs.dot(left_out_log_probs-left_out_log_disc)
            objective_disc = discretized_aux_kl + f_eval() + common_entropy_term
            
            if objective_disc < tolerance:
                bits_min = b
                break
        print(bits_min, file=sys.stderr)

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
Reads data from stdin: \"length frequency\\n\" format
(\"frequency length\\n\" if swap is on).
Writes the result to stdout and optimization info to stderr.
Author: Gábor Borbély, License: MIT
Contact: http://math.bme.hu/~borbely/indexeng""",
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

During the MDL evaluation the minimum number of bits are calculated
which encode the model within tolerance.
If the model is worse than the tolerance then "inf" is printed.
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
    
    parser.add_argument('-mae', "--mae", dest='mae', type=float, default=0,
                    help="The gradient descent stops if the Max Absolute Error of the gradient is below this threshold," + 
                         "if not positive then the MAE of the gradient is irrelevant.")
                    
    parser.add_argument('-e', "--eta", dest='eta', type=float, default=1,
                    help='learning rate')

    parser.add_argument("--opt", "--optimizer", dest='opt', type=str,
                    choices=["Adagrad", "GradientDescent"], default="Adagrad")                
    
    parser.add_argument('-l', "--load", dest='load', type=str, default="",
                    help='load model from file')
    
    parser.add_argument('-r', "--random", dest='random', default=False,
                    help='random initial model (otherwise uniform)', action='store_true')
                                        
    parser.add_argument('--eval', "--evaluate", dest='eval', action='store_true',
                    default=False, help='evaluate the learned model')

    parser.add_argument('-s', '--swap', dest='swap', default=False,
                    help='swap columns in input', action='store_true')

    parser.add_argument("--mdl", dest="mdl", type=float, default=0.0, metavar='tolerance',
                    help='perform MDL evaluation of the model if a positive number is set.')

    exit(main(parser.parse_args()))
