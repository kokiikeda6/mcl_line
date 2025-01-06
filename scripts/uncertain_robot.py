#!/usr/bin/env python3

import sys
sys.path.append('../scripts/')
from sim_robot import*
from scipy.stats import expon, norm, uniform

class Robot(SimRobot):
    def __init__(self, pose, agent=None, sensor=None, color="black", \
                 bias_rate_stds=0.1, \
                 expected_stack_time=1e100, expected_escape_time=1e-100, \
                 expected_kidnap_time=1e100, kidnap_range_x=(-5.0, 5.0)):
        super().__init__(pose, agent, sensor, color)
        self.bias_rate_nu =norm.rvs(loc=1.0, scale=bias_rate_stds)
        self.stuck_pdf = expon(scale=expected_stack_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape =self.escape_pdf.rvs()
        self.is_stuck = False
        self.kidnap_pdf = expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx = kidnap_range_x
        self.kidnap_dist = uniform(loc=rx[0], scale=rx[1]-rx[0])

    def bias(self, nu):
        return nu*self.bias_rate_nu
    
    def stuck(self, nu, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu*(not self.is_stuck)
    
    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return self.kidnap_dist.rvs()
        else:
            return pose
    
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu = self.agent.decision(obs)
        nu = self.bias(nu)
        nu = self.stuck(nu, time_interval)
        self.pose = self.state_transition(nu, time_interval, self.pose)
        self.pose = self.kidnap(self.pose, time_interval)

class Camera(SimCamera):
    def __init__(self, env_map, distance_range=(0.5, 4.0), \
                 distance_noise_rate=0.1, distance_bias_rate_stddev=0.1, \
                 phantom_prob=0.0, phantom_range_x=(-5.0, 5.0), \
                 oversight_prob=0.1, occlusion_prob=0.0):
        super().__init__(env_map, distance_range)
        
        self.distance_noise_rate = distance_noise_rate
        self.distance_bias_rate_std = norm.rvs(scale=distance_bias_rate_stddev)
        rx = phantom_range_x
        self.phantom_dist = uniform(loc=rx[0], scale=(rx[1]-rx[0]))
        self.phantom_prob = phantom_prob

        self.oversight_prob = oversight_prob
        self.occlusion_prob = occlusion_prob

    def noise(self, relpos):
        ell = norm.rvs(loc=relpos, scale=relpos*self.distance_noise_rate)
        return ell
    
    def bias(self, relpos):
        return relpos + relpos*self.distance_bias_rate_std
    
    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = self.phantom_dist.rvs()
            return self.observation_function(cam_pose, pos)
        else:
            return relpos
        
    def oversight(self, relpos):
        if uniform.rvs() < self.phantom_prob:
            return None
        else:
            return relpos
        
    def occlusion(self, relpos):
        if uniform.rvs() < self.occlusion_prob:
            ell = relpos + uniform.rvs()*(self.distance_range[1] - relpos)
            return ell
        else:
            return relpos

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z)
            z = self.occlusion(z)
            z = self.oversight(z)
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed

if __name__ == '__main__':
    world = World(30, 0.1)

    m = Map()
    m.append_landmark(Landmark(2))
    world.append(m)

    for i in range(1):
        straight = Agent(0.2)
        r = Robot(-2, sensor=Camera(m, occlusion_prob=0.1), agent=straight, color="gray")
        world.append(r)

    world.draw()