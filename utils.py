# -*- coding: utf-8 -*-
from __future__ import print_function
import sys

import math
import numpy

def logdet(M):
    x = numpy.linalg.slogdet(M)
    if x[0] > 0:
        return x[1]
    else:
        # print(x[0]*numpy.exp(x[1]), file=sys.stderr)
        return float("inf")

def read_stats(f, xmin=1, xmax=100, normalize=False, swap=False, dtype="float32"):
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
    y_data = data[:, 1].astype(dtype)
    
    factor = 1.0/y_data.sum() if normalize else 1.0
    return x_data, y_data * factor

def log_simplex_volume(d):
    return 0.5*numpy.log(d) - logfactorial(d-1)
    # return numpy.log(numpy.sqrt(d)/numpy.math.factorial(d-1))
def logfactorial(n):
    result = 0.0
    while n > 1:
        result += numpy.log(n)
        n-=1
    return result

def constraint_mtx(k, dtype="float32"):
    J = numpy.zeros((k, k-1), dtype=dtype)
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
            return (parts, KL+common_ent)
        return (parts[0] + parts[2]/n + (parts[1]/(2.0*n))*numpy.log(n/(2.0*numpy.pi)), KL+common_ent)
