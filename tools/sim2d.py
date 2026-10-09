"""Simulador 2D para pruebas offline (la geometria del escenario solo se usa aqui)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dock_lidar.perception import DockDetector
from dock_lidar.tracker import DockTracker
from dock_lidar.controller import DockController

RECTS = np.array([
    [1.95, 2.05, -2.0, 2.0],
    [-4.05, -3.95, -2.0, 2.0],
    [-4.0, 2.0, 1.95, 2.05],
    [-4.0, 2.0, -2.05, -1.95],
    [1.8695, 1.9495, 0.0475, 0.1275],
    [1.8695, 1.9495, -0.1275, -0.0475],
])
LASER = (-0.050502, -0.017960, 3.14)
N, AMIN, AINC = 720, -math.pi, 0.008738775737583637
R_ROBOT = 0.1695


def raycast(ox, oy, ang):
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        tx1 = (RECTS[:, 0] - ox) / dx; tx2 = (RECTS[:, 1] - ox) / dx
        ty1 = (RECTS[:, 2] - oy) / dy; ty2 = (RECTS[:, 3] - oy) / dy
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (tmax >= tmin) & (tmax > 0)
    t = np.where(hit, np.where(tmin > 0, tmin, np.inf), np.inf)
    return t.min(axis=1)


def collides(x, y):
    for xmin, xmax, ymin, ymax in RECTS:
        cx, cy = min(max(x, xmin), xmax), min(max(y, ymin), ymax)
        if (cx - x) ** 2 + (cy - y) ** 2 < R_ROBOT ** 2:
            return True
    return False


def is_docked(x, y, yaw):
    return (1.95 - x) <= 0.268 and abs(y) < 0.0077 and abs(yaw) < math.radians(6)


def run(x, y, yaw, tmax=180.0, noise=0.001, verbose=False, seed=1):
    rng = np.random.default_rng(seed)
    det, trk, ctl = DockDetector(), DockTracker(), DockController()
    dt, t, v, w = 0.05, 0.0, 0.0, 0.0
    collisions, last_state, pts = 0, None, None
    odom = np.array([0.0, 0.0, 0.0])
    while t < tmax:
        k = int(round(t / dt))
        docked = is_docked(x, y, yaw)
        if k % 2 == 0:
            lx = x + math.cos(yaw) * LASER[0] - math.sin(yaw) * LASER[1]
            ly = y + math.sin(yaw) * LASER[0] + math.cos(yaw) * LASER[1]
            ang = yaw + LASER[2] + AMIN + AINC * np.arange(N)
            r = raycast(lx, ly, ang) + rng.normal(0, noise, N)
            d = det.detect(r, AMIN, AINC, 0.15, 12.0, LASER)
            pts = det.last_points
            if d is not None:
                trk.update(d.center, d.normal, tuple(odom), t)
        est = trk.get(tuple(odom), t)
        v_cmd, w_cmd, info = ctl.step(est, tuple(odom), pts, docked, t)
        if ctl.state != last_state:
            if verbose:
                print(f'  t={t:6.2f}  {last_state} -> {ctl.state}  {info.get("reason", "")}')
            last_state = ctl.state
        if ctl.state == 'DOCKED':
            return dict(ok=True, t=t, lat=y, yaw=math.degrees(yaw), wall=1.95 - x,
                        coll=collisions, retries=ctl.retries)
        v += np.clip(v_cmd - v, -0.9 * dt, 0.9 * dt)
        w += np.clip(w_cmd - w, -3.0 * dt, 3.0 * dt)
        nx = x + v * math.cos(yaw) * dt
        ny = y + v * math.sin(yaw) * dt
        if abs(ny) < 0.15 and 1.95 - nx < 0.262:
            nx = 1.95 - 0.262
        if collides(nx, ny):
            collisions += 1
            nx, ny, v = x, y, 0.0
        dyaw = w * dt
        odom[0] += v * math.cos(odom[2]) * dt
        odom[1] += v * math.sin(odom[2]) * dt
        odom[2] += dyaw * 1.02
        x, y, yaw = nx, ny, (yaw + dyaw + math.pi) % (2 * math.pi) - math.pi
        t += dt
    return dict(ok=False, t=t, lat=y, yaw=math.degrees(yaw), wall=1.95 - x,
                coll=collisions, retries=ctl.retries, state=ctl.state)


POSES = [
    (0.4113, -0.1825, 0.3601),
    (0.9, 0.7, -1.2),
    (1.2, -1.2, 2.5),
    (1.5, 1.4, math.pi),
    (1.55, 0.35, 0.0),
    (-1.0, -1.5, 0.0),
    (0.0, 0.0, math.pi),
    (-3.0, 1.5, 2.0),
    (-3.5, -1.6, -2.5),
    (1.0, 0.0, 1.5708),
]

if __name__ == '__main__':
    if len(sys.argv) == 4:
        poses = [tuple(float(a) for a in sys.argv[1:])]
    else:
        poses = POSES
    ok = 0
    for P in poses:
        print(f'Pose inicial x={P[0]:.2f} y={P[1]:.2f} yaw={math.degrees(P[2]):.0f} deg')
        res = run(*P, verbose=True)
        ok += res['ok']
        print('  ->', 'ACOPLADO' if res['ok'] else 'FALLO', ' '.join(
            f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in res.items() if k != 'ok'))
    print(f'\nExitos: {ok}/{len(poses)}')
