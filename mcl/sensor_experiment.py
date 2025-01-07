#!/usr/bin/env python3

import sys
sys.path.append('..e/scripts/')
from robot import*

m = Map()
m.append_landmark(Landmark(1))

distance = []

for i in range(1000):
    c = Camera(m)
    d = c.data(0.0)
    if len(d) > 0:
        distance.append(d[0][0])
print(distance)
print(np.std(distance))