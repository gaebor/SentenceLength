from __future__ import print_function

import math
import sys
import argparse
from collections import Counter

def read_dist(filename, separator, flip=False):
    counter = Counter()

    for key, num in dist_reader(filename, separator, flip):
        counter[key] += num
    
    return dict(counter)

def dist_reader(filename, separator, flip=False):
    with open(filename) as f:
        for line in f:
            if not separator:
                one, two = line.strip().split()
                if flip:
                    num, key = one, two
                else:
                    key, num = one, two
            else:
                if flip:
                    num, key = line.strip().split(separator, 1)
                else:
                    key, num = line.strip().rsplit(separator, 1)
            num = float(num)
            if num > 0:
                yield key, num

def iterate_values(args):
    if args.sorted:
        iter1 = dist_reader(args.file1, args.separator, args.flip)
        iter2 = dist_reader(args.file2, args.separator, args.flip)
        left_from_1 = False
        left_from_2 = False
        try:
            next1 = next(iter1)
            left_from_1 = True
            next2 = next(iter2)
            left_from_2 = True
            while True:
                if next1[0] < next2[0]:
                    yield next1[1], 0.0
                    left_from_1 = False
                    next1 = next(iter1)
                    left_from_1 = True
                elif next1[0] > next2[0]:
                    yield 0.0, next2[1]
                    left_from_2 = False
                    next2 = next(iter2)
                    left_from_2 = True
                else:
                    yield next1[1], next2[1]
                    left_from_1 = False
                    left_from_2 = False
                    next1 = next(iter1)
                    left_from_1 = True
                    next2 = next(iter2)
                    left_from_2 = True
        except StopIteration:
            pass
        if left_from_1 and left_from_2:
            yield next1[1], next2[1]
        elif left_from_1:
            yield next1[1], 0.0
        elif left_from_2:
            yield 0.0, next2[1]
        # consume rest, if any
        for next1 in iter1:
            yield next1[1], 0.0
        for next2 in iter2:
            yield 0.0, next2[1]
    else:
        P = read_dist(args.file1, args.separator, args.flip)
        Q = read_dist(args.file2, args.separator, args.flip)
        suppP = set(P.keys())
        suppQ = set(Q.keys())
        for k in suppP - suppQ:
            yield P[k], 0.0
        for k in suppP & suppQ:
            yield P[k], Q[k]
        for k in suppQ - suppP:
            yield 0.0, Q[k]

def main(args):
    Pcommon = 0.0 # prob of common support according to P
    Qcommon = 0.0 # prob of common support according to Q
    
    sumP = 0.0
    sumQ = 0.0
    
    kl = 0.0 # - sum p*log(p/q)
    
    for p, q in iterate_values(args):
        if q == 0:
            sumP += p
        elif p == 0:
            sumQ += q
        else:
            kl += p*math.log(p/q)
            # kl2 += q*math.log(q/p)
            Pcommon += p
            Qcommon += q

    sumP = sumP + Pcommon if args.renormalizeP else 1.0
    sumQ = sumQ + Qcommon if args.renormalizeQ else 1.0

    if sumP == 0:
        return 1
    
    Pcommon /= sumP
    # Qcommon /= sumQ
    
    a1 = -Pcommon*math.log(Pcommon) if Pcommon > 0 else 0.0
    # a2 = -Qcommon*math.log(Qcommon) if Qcommon > 0 else 0.0
    
    b1 = (Pcommon*math.log(sumQ/sumP) if sumQ > 0 else 0.0) + kl/sumP
    # b2 = Qcommon*math.log(sumP/sumQ) + kl2/sumQ
    print(Pcommon, file=sys.stderr)
    print(a1 + b1)
    
    if Pcommon < math.exp(-1.0):
        return 1
    else:
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file1", help="first distribution, P", type=str)
    parser.add_argument("file2", help="second distribution, Q", type=str)
    parser.add_argument("-S", "--sorted", dest="sorted", action="store_true",
                        help="""tells whether distributions are sorted lexicographically.
 Its easier to calculate intersection this way.""")
    parser.add_argument("-s", "--separator", dest="separator", type=str, default="",
                        help="separator in input files")
    parser.add_argument("-f", "--flip", dest="flip", action="store_true",
                        help="if \"flip\", then columns are: \"number key\" otherwise \"key number\"")
    parser.add_argument("-P", dest="renormalizeP", action="store_true",
                        help="tells whether re-normalize P")
    parser.add_argument("-Q", dest="renormalizeQ", action="store_true",
                        help="tells whether re-normalize Q")
    
    exit(main(parser.parse_args()))
