from __future__ import print_function
import numpy
import numpy.random

# p = numpy.ones(3)/3.0
p = numpy.array([0.5, 0.25, 0.25])

sample_size = 500000
time = 1000
k = 3
steps = range(-1, len(p)-1)

for _ in range(sample_size):
    x = k
    i = 0
    while x>0 and i < time:
        x += numpy.random.choice(steps, p=p)
        i += 1
    if i < time: print(i)
