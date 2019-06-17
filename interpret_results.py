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
            if not simple:
                metrics = list(map(float, line[2:]))
                metrics.append(n)
                this_cost = cost(*metrics, tol=t)
            else:
                this_cost = mdl(int(line[1]), list(map(int, line[2].split('.'))), line[3])
        except:
            print(*line, file=sys.stderr)
            continue
        if data not in results or results[data][1] > this_cost:
                results[data] = [this_info, this_cost]
    for data in sorted(results):
        print(data, results[data][0])
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
