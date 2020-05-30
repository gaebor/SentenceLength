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
        if i < len(p):
            print("#", k[i], alpha[i], p[i].dot(steps), *p[i], file=file)
        else:
            print("#", k[i], alpha[i], file=file)
    if len(p) > 1:
        print("# average values", alpha.dot(p.dot(steps)), *alpha.dot(p), file=file)

    for i in range(len(x)):
        print(probs[i], x[i], file=file)

def load_model(file_name, dtype="float32"):
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
                    if len(line) > 3:
                        p.append([float(x) for x in line[3:]])
                except:
                    continue
        else:
            raise ValueError("not a model")
    
    return (numpy.array(k, dtype='int32'),
            numpy.array(alpha, dtype=dtype),
            numpy.array(p, dtype=dtype))

def diagonal(A, k=0):
    return A.reshape((-1,)+A.shape[2:])[k::A.shape[1]+1]

def main(args):
    learned = args.filename + ".o" + str(args.order) + (".c" if args.coupled else ".k") + ".".join(map(str, args.k)) + ".learned"
    try:
        k, alpha_initial, x_initial = load_model(learned, dtype="float64")
        alpha_initial = numpy.log(alpha_initial)
        x_initial = numpy.log(x_initial)
        if (k != numpy.array(args.k, dtype='int32')).any() or \
            x_initial.shape[1] != args.order + 2:
            print("Model \"" + learned + "\" has order", x_initial.shape[1]-2, "and k", k, file=sys.stderr)
            raise
        if len(alpha_initial) != len(k):
            print("Model \"" + learned + "\" has inconsistent number of mixture components: ", len(k), "!=", len(alpha_initial), file=sys.stderr)
            raise
        if args.coupled:
            if len(x_initial) != 1:
                print("Coupled model \"" + learned + "\" has too many mixture components: ", len(x_initial), "!= 1", file=sys.stderr)
                raise
        else:
            if len(x_initial) != len(alpha_initial):
                print("Non-coupled model \"" + learned + "\" has inconsistent number of mixture components: ", len(x_initial), "!=", len(alpha_initial), file=sys.stderr)
                raise
        order = args.order + 2
    except:
        order = args.order + 2 # span of steps
        k = numpy.sort(numpy.abs(numpy.array(args.k, dtype="int32")))
        if args.random:
            x_initial = numpy.random.rand(1 if args.coupled else len(k), order).astype("float64")
            alpha_initial = numpy.random.rand(len(k)).astype("float64")
        else:
            x_initial = numpy.zeros((1 if args.coupled else len(k), order), dtype="float64")
            alpha_initial = numpy.zeros(len(k), dtype="float64")
    
    k_min = min(k)
    k_max = max(k)
    
    if k_min <= 0:
        print("ERROR: k values should be positive integers!", file=sys.stderr)
        return 1

    x_data, y_data = read_stats(args.filename if args.filename else sys.stdin, 
                                args.xmin, args.xmax, normalize=True, 
                                swap=args.swap, dtype="float64")
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
    
    import tensorflow as tf
    
    # constants
    i_range = tf.range(x_min, x_max + 1, dtype="float64")
    i_range_c = tf.cast(i_range, "complex128")
    x_tf = tf.constant(x_common-x_min, dtype="int32")
    k_tf = tf.constant(k, dtype="float64")
    multiplier = k_tf[:, None]/i_range[None, :]
    y_tf = tf.constant(y_common)
    H_common = -y_common.dot(y_common)
    
    # variables
    x = tf.Variable(x_initial)
    alpha = tf.Variable(alpha_initial)
    
    # def objective(x):
    alpha_softmax = tf.nn.softmax(alpha)
    
    p = tf.nn.softmax(x, 1)
    
    print(alpha_softmax.numpy())
    print(p.numpy())
    
    # hopefully this padding is not too big, not too small
    p = tf.pad(p, [[0, 0], [0, (order-1)*x_max + 2 - p.shape[1]]])
    
    fp = tf.signal.rfft(p)
    fp_power = fp[:, None, :] ** i_range_c[None, :, None]
    # shape=(len(k), x_max-x_min+1, order*x_max)
    # TODO coupled
    powers = tf.signal.irfft(fp_power)
    for i in range(len(k)):
        print("k={}".format(k[i]))
        print(chop(powers[i].numpy()))
    if k_max > x_min:
        powers = tf.pad(powers, [[0, 0], [0, 0], [k_max-x_min, 0]])
    # shape=(len(k), x_max-x_min+1)
    probs = multiplier*tf.stack([diagonal(powers[i], k=x_min-k[i]) for i in range(len(k))], axis=0)
    
    print(probs.numpy())

    # shape=(len(k), len(x_common))
    generated_probs = tf.tensordot(alpha_softmax, tf.gather(probs, x_tf, axis=1), [0, 0])
    
    error = -tf.tensordot(y_tf, tf.math.log(generated_probs), 1) - H_common
    
    optimizer = eval("tf.keras.optimizers." + args.opt)(learning_rate=args.eta)
    
    # optimizer = eval("thextensions." + args.opt + "Optimizer")(error, x, alpha_, eta=args.eta)
    # optimizer.init(x_initial, alpha_initial)
    
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

    # number of model parameters
    d = p_learned.shape[0] * (p_learned.shape[1] - 1) + len(alpha_learned) - 1
    # number of auxiliary parameters
    d_aux = 0
    if left_out_dim > 0:
        d_aux = left_out_dim - 1
        common_entropy_term = -covered_weight*numpy.log(covered_weight) if covered_weight > 0 else 0.0
    else:
        common_entropy_term = 0.0

    # model volume
    model_volume = p_learned.shape[0] * log_simplex_volume(p_learned.shape[1]) + log_simplex_volume(len(alpha_learned))
            
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
        
    # (number of parameters) * (number of free parameters)
    J = numpy.zeros((p_learned.size + alpha_learned.size, d), dtype="float64")
    for i in range(len(p_learned)):
        J[i*order:(i+1)*order, i*(order-1):(i+1)*(order-1)] = constraint_mtx(order)
    if len(alpha_learned) > 1:
        # mixing coefficients
        J[-len(alpha_learned):, -(len(alpha_learned)-1):] = constraint_mtx(len(alpha_learned))
    hessian = logdet(J.transpose().dot(f_hessian()).dot(J))
    print(objective, common_entropy_term, model_volume, left_out_volume,
              hessian, left_out_hessian, d+d_aux, file=sys.stdout)

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

    parser.add_argument('-m', "--max", dest='xmax', type=int, default=1000,
                    help='maximum sentence length')
    
    parser.add_argument("-min", "--min", dest='xmin', type=int, default=1,
                    help='minimum sentence length')
                    
    parser.add_argument('-k', dest='k', type=int, nargs="+", default=[3],
                    help='k values in the mixture')
    
    parser.add_argument('-o', "--order", dest='order', type=int, default=2,
                    help='how many steps can be taken upwards')
    
    parser.add_argument('-i', "--iter", dest='iter', type=int, default=1000,
                    help="Maximum number of steps for the gradient descent")
    
    parser.add_argument('-mae', "--mae", dest='mae', type=float, default=1e-3,
                    help="The gradient descent stops if the Max Absolute Error of the gradient is below this threshold," + 
                         "if not positive then the MAE of the gradient is irrelevant.")

    parser.add_argument('-e', "--eta", dest='eta', type=float, default=0.5,
                    help='learning rate')

    parser.add_argument("--opt", "--optimizer", dest='opt', type=str,
                    choices=["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl"
                             "Nadam", "RMSprop", "SGD"], default="SGD")

    parser.add_argument("filename", type=str, default="",
                    help='data filename (stdin if empty), model filename is inferred')
    
    parser.add_argument('-r', "--random", dest='random', default=False,
                    help='random initial model (otherwise uniform)', action='store_true')

    parser.add_argument('-c', "--coupled", dest='coupled', default=False,
                    help='couple probabilities of different k values'+
                        '(if more than one k value is used)', action='store_true')

    parser.add_argument('-s', '--swap', dest='swap', default=False,
                    help='swap columns in input', action='store_true')
    
    # parser.add_argument('-d', '--dtype', dest='dtype', type=str, default="float32",
                    # help='float representation')
    
    exit(main(parser.parse_args()))
