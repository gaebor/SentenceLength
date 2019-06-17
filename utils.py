# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import argparse

import math
import numpy


from theano import config as tconfig

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
    y_data = data[:, 1].astype(tconfig.floatX)
    
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
                numpy.array(alpha, dtype=tconfig.floatX),
                numpy.array(p, dtype=tconfig.floatX))

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
    J = numpy.zeros((k, k-1), dtype=tconfig.floatX)
    for n in range(1, k):
        J[:n,n-1] = -1.0/numpy.sqrt(n*(n+1.0))
        J[n,n-1] = numpy.sqrt(n/(n+1.0))
    return J

def discretize(x, min_val, dx):
    return numpy.round((x - min_val)*dx)/dx + min_val

def ncosts(KL, common_ent, log_volume, log_aux_volume, hessian, aux_hessian, d):
        return KL+common_ent, d, log_volume+log_aux_volume + 0.5*hessian + 0.5*aux_hessian

def cost(KL, common_ent, log_volume, log_aux_volume, hessian, aux_hessian, d, n, tol=0):
        parts = ncosts(KL, common_ent, log_volume, log_aux_volume, hessian, aux_hessian, d)
        parts = (max(0, parts[0] - tol), parts[1], parts[2])
        if n == float("inf"):
            return parts
        return parts[0] + parts[2]/n + (parts[1]/(2.0*n))*numpy.log(n/(2.0*numpy.pi))
