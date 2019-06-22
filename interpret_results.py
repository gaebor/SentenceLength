from __future__ import print_function
import sys
import numpy

from utils import cost, ncosts

def disc_dim(x):
    return x if x > 1 else 0

def mdl(order, k, bits, type=0):
        if numpy.isfinite(float(bits)):
            return int(bits)*((order+2)*len(k) + disc_dim(len(k)) + disc_dim(k[0]-1))
        else:
            return float("inf")

def main(n, t, simple):
    results = {}

    for line in sys.stdin:
        line = line.strip().split()
        try:
            data, this_info = line[:2]
            metrics = list(map(float, line[2:]))
            if not simple:
                metrics.append(n)
                this_cost, fit = cost(*metrics, tol=t)
            else:
                this_cost, fit = (metrics[0], 0 if metrics[0] < float("inf") else 1)
        except:
            print(*line, file=sys.stderr)
            continue
        if data not in results or results[data][2] > this_cost:
            results[data] = [this_info, fit, this_cost]
    for data in sorted(results):
        print(data, *results[data])
    return 0

if __name__ == "__main__":
    n = None
    t = None
    if len(sys.argv) > 1:
        n = float(sys.argv[1])
        if len(sys.argv) > 2:
            t = float(sys.argv[2])
        else:
            t = 0.0
        simple = False
    else:
        simple = True
        
    exit(main(n, t, simple))
