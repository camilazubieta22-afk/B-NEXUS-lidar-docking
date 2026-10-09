"""Maquina de estados y control de aproximacion al dock."""
import math
from dataclasses import dataclass

import numpy as np

from .tracker import wrap, to_world, to_base


@dataclass
class ControlParams:
    v_max: float = 0.20
    v_final: float = 0.04
    w_max: float = 1.0
    k_w: float = 2.0
    k_v: float = 0.8
    rotate_in_place: float = 0.6
    lookahead_gain: float = 0.6
    lookahead_min: float = 0.12
    lookahead_max: float = 0.60
    dock_dist: float = 0.18
    predock_dist: float = 0.70
    predock_tol: float = 0.06
    final_zone: float = 0.40
    cone_slope: float = 0.6
    cone_margin: float = 0.04
    final_abort_lat: float = 0.04
    final_abort_head: float = 0.25
    final_check_dist: float = 0.30
    stall_time: float = 4.0
    backoff_speed: float = 0.10
    backoff_dist: float = 0.40
    search_time: float = 1.5
    explore_step: float = 1.2
    explore_timeout: float = 15.0
    robot_radius: float = 0.17
    safety_dist: float = 0.15


def axis_geometry(center, normal):
    n = normal / np.linalg.norm(normal)
    t = np.array([-n[1], n[0]])
    r = -np.asarray(center)
    along = float(r @ n)
    lat = float(r @ t)
    head = wrap(-math.atan2(-n[1], -n[0]))
    return along, lat, head


class DockController:
    def __init__(self, params=None):
        self.p = params or ControlParams()
        self.state = 'SEARCH'
        self.t_state = None
        self.retries = 0
        self.explore_goal = None
        self.backoff_start = None
        self.best_along = None
        self.t_progress = None

    def _set(self, s, t):
        if s != self.state:
            self.state = s
            self.t_state = t
            self.explore_goal = None
            self.best_along = None
            self.t_progress = t

    def _drive_to(self, P, v_cap):
        alpha = math.atan2(P[1], P[0])
        dist = float(np.hypot(P[0], P[1]))
        w = float(np.clip(self.p.k_w * alpha, -self.p.w_max, self.p.w_max))
        if abs(alpha) > self.p.rotate_in_place:
            return 0.0, w, alpha
        v = min(v_cap, self.p.k_v * dist) * max(0.0, math.cos(alpha))
        return v, w, alpha

    def _front_blocked(self, pts):
        if pts is None or len(pts) == 0:
            return False
        r = self.p.robot_radius
        m = (pts[:, 0] > 0) & (pts[:, 0] < r + self.p.safety_dist) & (np.abs(pts[:, 1]) < r + 0.03)
        return bool(m.any())

    def _rear_blocked(self, pts):
        if pts is None or len(pts) == 0:
            return False
        r = self.p.robot_radius
        m = (pts[:, 0] < 0) & (pts[:, 0] > -(r + 0.08)) & (np.abs(pts[:, 1]) < r)
        return bool(m.any())

    def _in_cone(self, along, lat, extra=0.0):
        return abs(lat) <= self.p.cone_slope * max(along - self.p.dock_dist, 0.0) \
            + self.p.cone_margin + extra

    def step(self, est, pose, pts, docked, t):
        p = self.p
        if self.t_state is None:
            self.t_state = t
            self.t_progress = t
        info = {}

        geo = None
        if est is not None:
            G, n, age = est
            along, lat, head = axis_geometry(G, n)
            geo = (G, n / np.linalg.norm(n), along, lat, head)
            info.update(along=along, lat=lat, head=head, age=age)

        if docked:
            self._set('DOCKED', t)
            return 0.0, 0.0, info
        if self.state == 'DOCKED':
            self._set('FINAL', t)

        if geo is None:
            if self.state == 'BACKOFF':
                return self._backoff(None, pose, pts, t, info)
            if self.state not in ('SEARCH', 'EXPLORE'):
                self._set('SEARCH', t)
            if self.state == 'SEARCH':
                if t - self.t_state > p.search_time:
                    self._set('EXPLORE', t)
                else:
                    return 0.0, 0.0, info
            return self._explore(pose, pts, t, info)

        G, n, along, lat, head = geo

        if self.state in ('SEARCH', 'EXPLORE'):
            self._set('APPROACH' if self._in_cone(along, lat) else 'GOTO_PREDOCK', t)

        if self.state == 'GOTO_PREDOCK':
            Q = G + n * p.predock_dist
            info['target'] = Q
            if np.hypot(Q[0], Q[1]) < p.predock_tol:
                self._set('APPROACH', t)
            else:
                v, w, _ = self._drive_to(Q, p.v_max)
                return v, w, info

        if self.state == 'APPROACH':
            if along < p.final_zone:
                self._set('FINAL', t)
            elif not self._in_cone(along, lat, extra=0.08):
                self._set('GOTO_PREDOCK', t)
                return 0.0, 0.0, info
            else:
                L = float(np.clip(p.lookahead_gain * (along - p.dock_dist),
                                  p.lookahead_min, p.lookahead_max))
                P = G + n * max(along - L, p.dock_dist - 0.05)
                info['target'] = P
                speed = p.v_max * float(np.clip((along - p.dock_dist) / 0.6, 0.3, 1.0))
                v, w, _ = self._drive_to(P, speed)
                v = min(v, speed)
                return v, w, info

        if self.state == 'FINAL':
            if along < p.final_check_dist and (abs(lat) > p.final_abort_lat
                                               or abs(head) > p.final_abort_head):
                info['reason'] = 'final torcido'
                self.retries += 1
                self._set('BACKOFF', t)
                return self._backoff(geo, pose, pts, t, info)
            if along > p.final_zone + 0.10:
                self._set('APPROACH', t)
                return 0.0, 0.0, info
            if self.best_along is None or along < self.best_along - 0.005:
                self.best_along = along
                self.t_progress = t
            elif t - self.t_progress > p.stall_time:
                info['reason'] = 'sin progreso'
                self.retries += 1
                self._set('BACKOFF', t)
                return self._backoff(geo, pose, pts, t, info)
            P = G + n * max(along - p.lookahead_min, p.dock_dist - 0.05)
            info['target'] = P
            alpha = math.atan2(P[1], P[0])
            w = float(np.clip(p.k_w * alpha, -p.w_max, p.w_max))
            speed = float(np.clip(0.4 * (along - p.dock_dist) + p.v_final, p.v_final, 0.08))
            v = 0.0 if abs(alpha) > 0.35 else speed * math.cos(alpha)
            return v, w, info

        if self.state == 'BACKOFF':
            return self._backoff(geo, pose, pts, t, info)

        return 0.0, 0.0, info

    def _backoff(self, geo, pose, pts, t, info):
        p = self.p
        if self.backoff_start is None or self.t_state == t:
            self.backoff_start = np.array(pose[:2])
        travelled = float(np.linalg.norm(np.array(pose[:2]) - self.backoff_start))
        done = travelled > p.backoff_dist
        w = 0.0
        if geo is not None:
            _, _, along, _, head = geo
            done = done or along > p.predock_dist * 0.8
            w = float(np.clip(-1.0 * head, -0.5, 0.5))
        if done or self._rear_blocked(pts) or t - self.t_state > 8.0:
            self.backoff_start = None
            self._set('GOTO_PREDOCK' if geo is not None else 'SEARCH', t)
            return 0.0, 0.0, info
        return -p.backoff_speed, w, info

    def _explore(self, pose, pts, t, info):
        p = self.p
        if self.explore_goal is None:
            if pts is None or len(pts) < 10:
                return 0.0, 0.0, info
            lo = np.percentile(pts, 3, axis=0)
            hi = np.percentile(pts, 97, axis=0)
            centre = 0.5 * (lo + hi)
            if np.hypot(*centre) < 0.5:
                far = pts[np.argmax(np.hypot(pts[:, 0], pts[:, 1]))]
                centre = far / np.linalg.norm(far) * p.explore_step
            self.explore_goal = to_world(pose, centre)
        goal_b = to_base(pose, self.explore_goal)
        info['target'] = goal_b
        if (np.hypot(*goal_b) < 0.15 or t - self.t_state > p.explore_timeout):
            self._set('SEARCH', t)
            return 0.0, 0.0, info
        v, w, alpha = self._drive_to(goal_b, p.v_max)
        if v > 0 and self._front_blocked(pts):
            self._set('SEARCH', t)
            return 0.0, 0.0, info
        return v, w, info
