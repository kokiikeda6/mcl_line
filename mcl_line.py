#!/usr/bin/env python3

import sys
sys.path.append('./scripts/')
from uncertain_robot import*
from scipy.stats import multivariate_normal
import math

class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose

    def motion_update(self, nu, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs()
        noised_nu = nu + ns*math.sqrt(abs(nu)/time)
        self.pose = SimRobot.state_transition(noised_nu, time, self.pose)

class Mcl:
    def __init__(self, init_pose, num, motion_noise_stds):
        self.particles = [Particle(init_pose) for i in range(num)]

        v = motion_noise_stds
        c = v["nn"]**2
        self.motion_noise_rate_pdf = multivariate_normal(mean=[0], cov=c)

    def motion_update(self, nu, time):
        for p in self.particles: p.motion_update(nu, time, self.motion_noise_rate_pdf)

    def draw(self, ax, elems):
        xs = [p.pose for p in self.particles]
        elems.append(ax.quiver(xs, [0]*len(xs), [1]*len(xs), [0]*len(xs), color="blue", alpha=0.5))

class EstimationAgent(Agent):
    def __init__(self, time_interval, nu, estimator):
        super().__init__(nu)
        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0.0

    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.time_interval)
        self.prev_nu = self.nu
        return self.nu

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)

def trial(motion_noise_stds):
    time_interval =0.1
    world = World(30, time_interval)
    initial_pose = -2 #x座標
    estimator = Mcl(initial_pose, 100, motion_noise_stds)
    straight = EstimationAgent(time_interval, 0.2, estimator)
    r = Robot(initial_pose, sensor=None, agent=straight)
    world.append(r)

    world.draw()

trial({"nn":0.18})

# m = Map()
# m.append_landmark(Landmark(2))
# world.append(m)


# world.draw()