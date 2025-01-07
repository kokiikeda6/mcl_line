#!/usr/bin/env python3

import sys
sys.path.append('../scripts/')
from uncertain_robot import*
from scipy.stats import multivariate_normal
import math
import random
import copy

class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight

    def motion_update(self, nu, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs()
        noised_nu = nu + ns*math.sqrt(abs(nu)/time)
        self.pose = SimRobot.state_transition(noised_nu, time, self.pose)

    def observation_update(self, observation, envmap, distance_dev_rate):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = SimCamera.observation_function(self.pose, pos_on_map)

            distance_dev = distance_dev_rate*particle_suggest_pos
            cov = distance_dev**2
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)

class Mcl:
    def __init__(self, envmap, init_pose, num, motion_noise_stds={"nn":0.19}, distance_dev_rate=0.14):
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate

        v = motion_noise_stds
        c = v["nn"]**2
        self.motion_noise_rate_pdf = multivariate_normal(mean=[0], cov=c)

    def motion_update(self, nu, time):
        for p in self.particles: p.motion_update(nu, time, self.motion_noise_rate_pdf)

    def observation_update(self, observation):
        for p in self.particles: p.observation_update(observation, self.map, self.distance_dev_rate)
        self.resampling()

    def resampling(self):
        ws = np.cumsum([e.weight for e in self.particles])
        if ws[-1] < 1e-100: ws = [e + 1e-100 for e in ws]

        step = ws[-1]/len(self.particles)
        r = np.random.uniform(0.0, step)
        cur_pos = 0
        ps = []

        while(len(ps) < len(self.particles)):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1
        
        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles: p.weight = 1.0/len(self.particles)

    def draw(self, ax, elems):
        xs = [p.pose for p in self.particles]
        vxs = [p.weight*len(self.particles) for p in self.particles]
        elems.append(ax.quiver(xs, [0]*len(self.particles), vxs, [0]*len(self.particles), angles='xy', scale_units='xy', scale=1.5, color="blue", alpha=0.5))

class EstimationAgent(Agent):
    def __init__(self, time_interval, nu, estimator):
        super().__init__(nu)
        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0.0

    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.time_interval)
        self.prev_nu = self.nu
        self.estimator.observation_update(observation)
        return self.nu

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)

def trial():
    time_interval = 0.1 # 時間間隔 [s]
    num_particles = 100 # パーティクルの数
    initial_pose = -4   # 初期のx座標
    velocity = 0.2      # ロボットの速度

    world = World(40, time_interval, debug=False)

    m = Map()
    m.append_landmark(Landmark(3))
    world.append(m)

    estimator = Mcl(m, initial_pose, num_particles)
    a = EstimationAgent(time_interval, velocity, estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()

trial()