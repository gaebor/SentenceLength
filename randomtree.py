# -*- coding: utf-8 -*-
from ast import literal_eval
import numpy, torch, torch.nn
from utils import *

def load_model(filename, dtype="float64"):
    with open(filename, "r") as f:
        data = literal_eval(f.read())
    x = numpy.array(data[0], dtype=dtype)
    return x[:3], x[3:], list(data[1].keys()), numpy.array(list(data[1].values()), dtype=dtype)

def write_model(p1, p2, k, t, filename):
    with open(filename, "w") as f:
        f.write(repr([p1.tolist() + p2.tolist(), dict(zip(k, t))]))

def P1(p1, N):
    M = torch.zeros((N+1,)*3)
    M[0,0,0] = 1
    pp = torch.nn.functional.softmax(p1, dim=0)
    M[1,0,0] = pp[0]
    M[0,1,0] = pp[1]
    M[0,0,1] = pp[2]
    for n in range(2, N+1):
        for i in range(0, n+1):
            for j in range(0, n+1-i):
                k = n-i-j
                if i > 0:
                    M[i,j,k] += M[i-1,j,k]*pp[0]
                if j > 0:
                    M[i,j,k] += M[i,j-1,k]*pp[1]
                if k > 0:
                    M[i,j,k] += M[i,j,k-1]*pp[2]
    return M

def P2(p2, N):
    M = torch.zeros((N+1,)*3)
    M[0,0,0] = 1
    pp = torch.nn.functional.softmax(p2, dim=0)
    # p200, p201, p220, p211, p212, p222
    # x^2    xy    xz    y^2   yz   z^2
    M[2,0,0] = pp[0]
    M[1,1,0] = pp[1]
    M[1,0,1] = pp[2]
    M[0,2,0] = pp[3]
    M[0,1,1] = pp[4]
    M[0,0,2] = pp[5]
    for n in range(4, N+1, 2):
        for i in range(0, n+1):
            for j in range(0, n+1-i):
                k = n-i-j
                # {0, 0, 2}, {0, 1, 1}, {0, 2, 0}, {1, 0, 1}, {1, 1, 0}, {2, 0, 0}
                if i > 1:
                    M[i,j,k] += M[i-2,j,k]*pp[0]
                if i > 0 and j > 0:
                    M[i,j,k] += M[i-1,j-1,k]*pp[1]
                if i > 0 and k > 0:
                    M[i,j,k] += M[i-1,j,k-1]*pp[2]
                if j > 1:
                    M[i,j,k] += M[i,j-2,k]*pp[3]
                if j > 0 and k > 0:
                    M[i,j,k] += M[i,j-1,k-1]*pp[4]
                if k > 1:
                    M[i,j,k] += M[i,j,k-2]*pp[5]
    return M

def fn(k, n, P1, P2):
    nnz = n > 0
    kappa = sum(k)
    result = torch.scalar_tensor(0)
    if n[0] - n[2] == kappa and n[0] >= k[0] and n[1] >= k[1] and n[2] >= k[2]:
        for i10 in range(max(k[0]-n[0], -n[1]), 1):
            for i11 in range(max(k[1], -i10), min(n[1], n[2]-k[2]-i10)+1):
                result += numpy.linalg.det(numpy.array([[n[0],0,0],[i10, i11, -i10-i11],[-i10+k[0]-n[0], -i11+k[1], i10+i11+k[2]]])[nnz][:, nnz])* \
                             P1[-i10, n[1] - i11, i10 + i11]*P2[i10 - k[0] + n[0], i11 - k[1], n[2] - (i10 + i11 + k[2])]
    return result/n[nnz].prod()

def f(k, n, P1, P2):
    kappa = sum(k)
    result = torch.scalar_tensor(0)
    for n0 in range(kappa, (kappa+n)//2 + 1):
        result += fn(k, numpy.array([n0, kappa + n - 2*n0, n0 - kappa]), P1, P2)
    return result

def F(p1, p2, x, kk):
    result = torch.zeros((len(x), len(kk)))
    P1M = P1(p1, max(x))
    P2M = P2(p2, max(x))
    for i, n in enumerate(x):
        for j, k in enumerate(kk):
            result[i, j] = f(k, n, P1M, P2M)
    return result

def main(args):
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.get_default_dtype()

    learned = args.filename + ".tree.learned"
    try:
        p1, p2, kk, t = load_model(learned, dtype="float64")
        p1 = numpy.log(p1)
        p2 = numpy.log(p2)
        t = numpy.log(t)
    except:
        p1 = torch.zeros(3)
        p2 = torch.zeros(6)
        kk = [(0,1,0), (0,0,1)]
        t = torch.zeros(len(kk))
    
    x, y = read_stats(args.filename, args.xmin, args.xmax, normalize=True, 
                                swap=args.swap, dtype="float64")
    y = torch.tensor(y)
    
    optimizer = eval("torch.optim." + args.opt)(lr=args.eta)

    if args.iter > 0:
        digits = math.ceil(math.log10(args.iter + 1))
        formatstr = "\t{:0{}d}\t{:.4e}\t{:.4e}"

        mae = float('inf')
        i = 1
        print("\titer\tobjective\tMAE", file=sys.stderr)
        while mae > args.mae and i <= args.iter:
            optimizer.zero_grad()
            tt = torch.nn.functional.softmax(t, dim=0)
            objective = y.dot(numpy.log(y/F(p1, p2, x, kk).matmul(tt)))
            objective.backward()
            optimizer.step()
            print(formatstr.format(i, digits, objective, mae), file=sys.stderr)
            i += 1
            if not numpy.isfinite(objective):
                break
        tt = torch.nn.functional.softmax(t, dim=0)
        p1 = torch.nn.functional.softmax(p1, dim=0)
        p2 = torch.nn.functional.softmax(p2, dim=0)

        write_model(p1, p2, kk, tt, learned)
        
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
