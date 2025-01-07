#!/usr/bin/env python3

import sys
sys.path.append('../scripts/')
from uncertain_robot import*
import math

world = World(40.0, 0.1)

initial_pose = 0
robots = []

for i in range(100):
    r = Robot(initial_pose, sensor=None, agent=Agent(0.1))
    world.append(r)
    robots.append(r)

world.draw()

poses = [abs(r.pose) for r in robots]
print(math.sqrt(np.var(poses)/np.mean(poses)))