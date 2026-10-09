"""Filtro temporal de la deteccion del dock en el frame odom."""
import math
from dataclasses import dataclass

import numpy as np


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def to_world(pose, p):
    return np.array(pose[:2]) + rot(pose[2]) @ p


def to_base(pose, p):
    return rot(pose[2]).T @ (np.asarray(p) - np.array(pose[:2]))


@dataclass
class TrackerParams:
    alpha: float = 0.5
    gate_dist: float = 0.25
    gate_angle: float = 0.35
    switch_count: int = 3
    timeout: float = 3.0


class DockTracker:
    def __init__(self, params=None):
        self.p = params or TrackerParams()
        self.reset()

    def reset(self):
        self.c = None
        self.n = None
        self.t_last = -1e9
        self.outliers = 0
        self.updates = 0

    def update(self, center_b, normal_b, pose, t):
        c_o = to_world(pose, center_b)
        n_o = rot(pose[2]) @ normal_b
        if self.c is None or (t - self.t_last) > self.p.timeout:
            self.c, self.n = c_o, n_o
            self.outliers, self.updates = 0, 1
            self.t_last = t
            return True
        dc = np.linalg.norm(c_o - self.c)
        da = abs(wrap(math.atan2(n_o[1], n_o[0]) - math.atan2(self.n[1], self.n[0])))
        if dc < self.p.gate_dist and da < self.p.gate_angle:
            a = self.p.alpha
            self.c = (1 - a) * self.c + a * c_o
            n = (1 - a) * self.n + a * n_o
            self.n = n / np.linalg.norm(n)
            self.outliers = 0
            self.updates += 1
            self.t_last = t
            return True
        self.outliers += 1
        if self.outliers >= self.p.switch_count:
            self.c, self.n = c_o, n_o
            self.outliers, self.updates = 0, 1
            self.t_last = t
            return True
        return False

    def get(self, pose, t):
        if self.c is None or (t - self.t_last) > self.p.timeout:
            return None
        c_b = to_base(pose, self.c)
        n_b = rot(pose[2]).T @ self.n
        return c_b, n_b, t - self.t_last
