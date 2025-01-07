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
        ax.plot([-5, 5], [0, 0], linestyle="--", alpha=0.5)

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

class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor

    def draw(self, ax, elems):
        x = self.pose
        xn = x + self.r 
        elems += ax.plot([x, xn], [0, 0], color="blue")
        c = patches.Circle(xy=(x, 0), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))
        self.poses.append(self.pose)
        elems += ax.plot([e for e in self.poses], [0]*len(self.poses), linewidth=0.5, color="black")
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)

    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu = self.agent.decision(obs)
        self.pose = self.state_transition(nu, time_interval, self.pose)

    @classmethod
    def state_transition(cls, nu, time, pose):
        return pose + nu * time
    
class Agent:
    def __init__(self, nu):
        self.nu = nu

    def decision(self, observation=None):
        return self.nu

class Landmark:
    def __init__(self, x):
        self.pos = x
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos, 0, s=100, marker="*", label="Landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos, 0, "id:"+str(self.id), fontsize=10))

class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks: lm.draw(ax, elems)

class IdealCamera:
    def __init__(self, env_map, distance_range=(0.5, 4.0)):
        self.map = env_map
        self.lastdata = []
        self.distance_range = distance_range

    def visible(self, polarpos):
        if polarpos is None:
            return False
        return self.distance_range[0] <= polarpos <= self.distance_range[1]


    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))
        
        self.lastdata = observed
        return observed

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x = cam_pose
            distance = lm[0]
            lx = x + distance
            elems += ax.plot([x,lx], [0, 0], color="pink")

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        distance = obj_pos - cam_pose
        return distance

if __name__ == '__main__':
    world = World(30, 0.1) # (シミュレート時間[s], Δt[s])

    m = Map()
    m.append_landmark(Landmark(2))
    world.append(m)

    straight = Agent(0.2)
    robot1 = IdealRobot(-2, sensor=IdealCamera(m), agent=straight)
    world.append(robot1)

    world.draw()