from __future__ import print_function
import sys
import numpy

def ncosts(KL, common_ent, log_volume, log_aux_volume, hessian, aux_hessian, d):
        return KL+common_ent, d, log_volume+log_aux_volume + 0.5*hessian + 0.5*aux_hessian

# def costs(KL, common_ent, log_volume, log_aux_volume, hessian, aux_hessian, d, n):
        # return KL, common_ent, log_volume/n, log_aux_volume/n, \
               # 0.5*hessian/n, 0.5*aux_hessian/n, (d/(2.0*n))*numpy.log(n/(2.0*numpy.pi))

def cost(KL, common_ent, log_volume, log_aux_volume, hessian, aux_hessian, d, n, tol=0):
        parts = ncosts(KL, common_ent, log_volume, log_aux_volume, hessian, aux_hessian, d)
        parts = (max(0, parts[0] - tol), parts[1], parts[2])
        if n == float("inf"):
            return parts
        return parts[0] + parts[2]/n + (parts[1]/(2.0*n))*numpy.log(n/(2.0*numpy.pi))

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
            data, order, k = line[:3]
            if not simple:
                metrics = list(map(float, line[3:]))
                metrics.append(n)
                this_cost = cost(*metrics, tol=t)
                this_info = []
            else:
                this_cost = mdl(int(line[1]), list(map(int, line[2].split('.'))), line[3])
                this_info = line[3:4]
        except:
            print(*line, file=sys.stderr)
            continue
        if data not in results or results[data][2] > this_cost:
                results[data] = [order, k, this_cost, this_info]
    for data in sorted(results):
        print(data, *(results[data][:3] + results[data][3]))
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
