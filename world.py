#!/usr/bin/env python3

import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np

class World:
    def __init__(self):
        self.objects = []

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        for obj in self.objects: obj.draw(ax)
        plt.show()

class SimRobot:
    def __init__(self, pose, color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color

    def draw(self, ax):
        x = self.pose
        xn = x + self.r
        ax.plot([x, xn], [0, 0], color="blue")
        c = patches.Circle(xy=(x, 0), radius=self.r, fill=False, color=self.color)
        ax.add_patch(c)

world = World()
robot1 = SimRobot(2)
world.append(robot1)
world.draw()