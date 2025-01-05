#!/usr/bin/env python3

import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_inrterval = time_interval

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        elems = []

        if self.debug:
            for i in range(1000): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=int(self.time_span/self.time_inrterval), interval=int(self.time_inrterval*1000), repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        time_str = "t = %.2f[s]"%(self.time_inrterval*i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_inrterval)

class SimRobot:
    def __init__(self, pose, agent=None, color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose]

    def draw(self, ax, elems):
        x = self.pose
        xn = x + self.r 
        elems += ax.plot([x, xn], [0, 0], color="blue")
        c = patches.Circle(xy=(x, 0), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        elems += ax.plot([e for e in self.poses], [0]*len(self.poses), linewidth=0.5, color="black")

    def one_step(self, time_interval):
        if not self.agent: return
        nu = self.agent.decision()
        self.pose = self.state_transition(nu, time_interval, self.pose)

    @classmethod
    def state_transition(cls, nu, time, pose):
        return pose + nu * time
    
class Agent:
    def __init__(self, nu):
        self.nu = nu

    def decision(self, observation=None):
        return self.nu

world = World(10,1) # (シミュレート時間[s], Δt[s])
straight = Agent(0.2)
robot1 = SimRobot(2, straight)
robot2 = SimRobot(2, color="red")
world.append(robot1)
world.append(robot2)
world.draw()