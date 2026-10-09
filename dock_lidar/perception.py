"""Deteccion de las cajas marcadoras en el LaserScan y calculo del eje del dock."""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PerceptionParams:
    box_width: float = 0.08
    box_protrusion: float = 0.08
    gap_width: float = 0.095
    max_range: float = 4.5
    ransac_iters: int = 150
    line_tol: float = 0.015
    min_line_points: int = 20
    min_line_length: float = 0.40
    max_lines: int = 6
    protrusion_tol: float = 0.03
    size_tol: float = 0.03
    min_box_points: int = 2
    seed: int = 0


@dataclass
class Cluster:
    s_lo: float
    s_hi: float
    n: int
    spacing: float

    @property
    def mid(self):
        return 0.5 * (self.s_lo + self.s_hi)

    @property
    def width(self):
        return self.s_hi - self.s_lo


@dataclass
class Detection:
    center: np.ndarray
    normal: np.ndarray
    wall_p0: np.ndarray
    wall_u: np.ndarray
    wall_s: tuple
    boxes: list
    box_points: np.ndarray
    widths: tuple
    gap: float
    separation: float
    protrusion: float
    gap_wall_points: int
    error: float
    box_point_count: int = 0

    def endpoints(self, i):
        s_lo, s_hi = self.boxes[i]
        base = self.wall_p0 + self.protrusion * self.normal
        return base + s_lo * self.wall_u, base + s_hi * self.wall_u


def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max,
                   laser_x, laser_y, laser_yaw, max_range):
    r = np.asarray(ranges, dtype=float)
    a = angle_min + angle_increment * np.arange(r.size)
    valid = np.isfinite(r) & (r > range_min) & (r < min(range_max, max_range))
    r = r[valid]
    a = a[valid] + laser_yaw
    x = laser_x + r * np.cos(a)
    y = laser_y + r * np.sin(a)
    pts = np.column_stack((x, y))
    order = np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))
    return pts[order]


def fit_line(pts):
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    _, v = np.linalg.eigh(cov)
    u = v[:, 1]
    n = np.array([-u[1], u[0]])
    return c, u, n


def extract_lines(pts, p, rng):
    lines = []
    remaining = np.arange(len(pts))
    for _ in range(p.max_lines):
        P = pts[remaining]
        m = len(P)
        if m < p.min_line_points:
            break
        i0 = rng.integers(0, m, size=p.ransac_iters)
        off = rng.integers(4, 25, size=p.ransac_iters)
        i1 = (i0 + off) % m
        a, b = P[i0], P[i1]
        d = b - a
        L = np.hypot(d[:, 0], d[:, 1])
        ok = L > 0.05
        if not ok.any():
            break
        a, d, L = a[ok], d[ok], L[ok]
        nrm = np.column_stack((-d[:, 1], d[:, 0])) / L[:, None]
        c = np.einsum('ij,ij->i', nrm, a)
        dist = np.abs(P @ nrm.T - c)
        best = int(np.argmax((dist < p.line_tol).sum(axis=0)))
        inl = dist[:, best] < p.line_tol
        if inl.sum() < p.min_line_points:
            break
        c0, _, n = fit_line(P[inl])
        inl = np.abs((P - c0) @ n) < p.line_tol
        if inl.sum() < p.min_line_points:
            break
        c0, u, _ = fit_line(P[inl])
        s = (P[inl] - c0) @ u
        if s.max() - s.min() >= p.min_line_length:
            lines.append(remaining[inl])
        remaining = remaining[~inl]
    return lines


def _clusters_along(s, spacing):
    clusters = []
    start = 0
    for k in range(1, len(s) + 1):
        if k == len(s) or (s[k] - s[k - 1]) > np.clip(2.5 * spacing[k], 0.02, 0.06):
            clusters.append(Cluster(float(s[start]), float(s[k - 1]), k - start,
                                    float(np.median(spacing[start:k]))))
            start = k
    return clusters


def match_signature(pts, line_idx, p, angle_inc):
    W = pts[line_idx]
    c0, u, n = fit_line(W)
    if np.dot(-c0, n) < 0:
        n = -n
    u = np.array([-n[1], n[0]])
    sw = (W - c0) @ u
    s_min, s_max = float(sw.min()), float(sw.max())

    rel = pts - c0
    s = rel @ u
    d = rel @ n
    band = ((np.abs(d - p.box_protrusion) < p.protrusion_tol)
            & (s > s_min - 0.2) & (s < s_max + 0.2))
    idx = np.nonzero(band)[0]
    if idx.size < 2 * p.min_box_points:
        return None

    idx = idx[np.argsort(s[idx])]
    ss = s[idx]
    r = np.hypot(pts[idx, 0], pts[idx, 1])
    cos_inc = np.abs((pts[idx] / r[:, None]) @ n)
    spacing = r * angle_inc / np.maximum(cos_inc, 0.3)
    clusters = _clusters_along(ss, spacing)

    bw, gw = p.box_width, p.gap_width
    best = None
    for c1, c2 in zip(clusters[:-1], clusters[1:]):
        if c1.n < p.min_box_points or c2.n < p.min_box_points:
            continue
        sp = max(c1.spacing, c2.spacing)
        tol = p.size_tol + sp
        gap = c2.s_lo - c1.s_hi
        sep = c2.mid - c1.mid
        e_w1 = abs(c1.width - (bw - sp))
        e_w2 = abs(c2.width - (bw - sp))
        e_gap = abs(gap - (gw + sp))
        e_sep = abs(sep - (bw + gw))
        if max(e_w1, e_w2, e_gap, e_sep) > tol:
            continue
        err = e_w1 + e_w2 + e_gap + e_sep
        if best is None or err < best[0]:
            best = (err, c1, c2)
    if best is None:
        return None

    err, c1, c2 = best
    in_boxes = ((ss >= c1.s_lo) & (ss <= c1.s_hi)) | ((ss >= c2.s_lo) & (ss <= c2.s_hi))
    box_pts = pts[idx[in_boxes]]
    protrusion = float(np.median(d[idx[in_boxes]]))
    gap_wall = int(np.sum((sw > c1.s_hi) & (sw < c2.s_lo)))
    center_s = 0.25 * (c1.s_lo + c1.s_hi + c2.s_lo + c2.s_hi)
    center = c0 + center_s * u + protrusion * n

    return Detection(center=center, normal=n, wall_p0=c0, wall_u=u,
                     wall_s=(s_min, s_max),
                     boxes=[(c1.s_lo, c1.s_hi), (c2.s_lo, c2.s_hi)],
                     box_points=box_pts, widths=(c1.width, c2.width),
                     gap=c2.s_lo - c1.s_hi, separation=c2.mid - c1.mid,
                     protrusion=protrusion, gap_wall_points=gap_wall,
                     error=float(err), box_point_count=int(len(box_pts)))


class DockDetector:
    def __init__(self, params=None):
        self.p = params or PerceptionParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.last_points = np.zeros((0, 2))
        self.last_lines = []

    def detect(self, ranges, angle_min, angle_increment, range_min, range_max,
               laser_pose):
        lx, ly, lyaw = laser_pose
        pts = scan_to_points(ranges, angle_min, angle_increment, range_min,
                             range_max, lx, ly, lyaw, self.p.max_range)
        self.last_points = pts
        self.last_lines = []
        if len(pts) < self.p.min_line_points:
            return None
        lines = extract_lines(pts, self.p, self.rng)
        best = None
        for li in lines:
            c0, u, _ = fit_line(pts[li])
            self.last_lines.append((c0, u, ((pts[li] - c0) @ u).min(),
                                    ((pts[li] - c0) @ u).max()))
            det = match_signature(pts, li, self.p, angle_increment)
            if det is not None and (best is None or det.error < best.error):
                best = det
        return best
