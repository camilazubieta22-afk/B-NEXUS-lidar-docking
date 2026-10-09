"""Nodo ROS 2: /scan -> deteccion -> control -> /cmd_vel."""
import json
import math
from dataclasses import fields

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener

from .perception import DockDetector, PerceptionParams
from .tracker import DockTracker, TrackerParams
from .controller import DockController, ControlParams

try:
    from irobot_create_msgs.msg import DockStatus
except ImportError:
    DockStatus = None


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class DockNode(Node):
    def __init__(self):
        super().__init__('dock_lidar')
        self.pp = self._declare_group('perception', PerceptionParams())
        self.tp = self._declare_group('tracker', TrackerParams())
        self.cp = self._declare_group('control', ControlParams())
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.control_rate = self.declare_parameter('control_rate', 20.0).value
        self.enabled = self.declare_parameter('enabled', True).value
        self.laser_fallback = (
            self.declare_parameter('laser_fallback_x', -0.050502).value,
            self.declare_parameter('laser_fallback_y', -0.017960).value,
            self.declare_parameter('laser_fallback_yaw', 3.14).value)

        self.detector = DockDetector(self.pp)
        self.tracker = DockTracker(self.tp)
        self.controller = DockController(self.cp)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.laser_pose = None
        self.dr_pose = np.zeros(3)
        self.use_dr = False
        self.last_cmd = (0.0, 0.0)
        self.last_det = None
        self.last_scan_t = None
        self.docked = False
        self.t_start = None
        self.finished_logged = False
        self.n_scans = 0
        self.n_dets = 0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mk_pub = self.create_publisher(MarkerArray, '/dock_lidar/markers', 10)
        self.st_pub = self.create_publisher(String, '/dock_lidar/state', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        if DockStatus is not None:
            self.create_subscription(DockStatus, '/dock_status', self.dock_cb,
                                     qos_profile_sensor_data)
        else:
            self.get_logger().warn('irobot_create_msgs no disponible: sin /dock_status')
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_cb)
        self.get_logger().info('dock_lidar listo. Esperando /scan ...')

    def _declare_group(self, prefix, obj):
        for f in fields(obj):
            val = self.declare_parameter(f'{prefix}.{f.name}', getattr(obj, f.name)).value
            setattr(obj, f.name, type(getattr(obj, f.name))(val))
        return obj

    def _lookup_laser(self, frame):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
            tr = tf.transform
            self.laser_pose = (tr.translation.x, tr.translation.y, yaw_from_quat(tr.rotation))
            self.get_logger().info(
                f'Montaje del LiDAR desde TF: x={self.laser_pose[0]:.4f} '
                f'y={self.laser_pose[1]:.4f} yaw={math.degrees(self.laser_pose[2]):.1f} deg')
        except Exception as e:
            self.get_logger().warn(f'TF {self.base_frame}<-{frame} no disponible ({e}); '
                                   'uso el montaje por defecto del README', throttle_duration_sec=5.0)
            return self.laser_fallback
        return self.laser_pose

    def _odom_pose(self, stamp=None):
        if not self.use_dr:
            try:
                t = Time() if stamp is None else Time.from_msg(stamp)
                tf = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, t,
                                                     timeout=Duration(seconds=0.05))
                tr = tf.transform
                return (tr.translation.x, tr.translation.y, yaw_from_quat(tr.rotation))
            except Exception:
                if stamp is not None:
                    return self._odom_pose(None)
                if self.n_scans > 30:
                    self.get_logger().warn('Sin TF odom->base_link: uso dead-reckoning')
                    self.use_dr = True
        return tuple(self.dr_pose)

    def dock_cb(self, msg):
        self.docked = bool(msg.is_docked)

    def scan_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.n_scans += 1
        self.last_scan_t = now
        if self.t_start is None:
            self.t_start = now
        if self.laser_pose is None:
            lp = self._lookup_laser(msg.header.frame_id)
        else:
            lp = self.laser_pose
        det = self.detector.detect(msg.ranges, msg.angle_min, msg.angle_increment,
                                   msg.range_min, msg.range_max, lp)
        self.last_det = det
        if det is not None:
            self.n_dets += 1
            pose = self._odom_pose(msg.header.stamp)
            self.tracker.update(det.center, det.normal, pose, now)

    def control_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 1.0 / self.control_rate
        v0, w0 = self.last_cmd
        self.dr_pose[0] += v0 * math.cos(self.dr_pose[2]) * dt
        self.dr_pose[1] += v0 * math.sin(self.dr_pose[2]) * dt
        self.dr_pose[2] += w0 * dt

        if self.last_scan_t is None:
            return
        if now - self.last_scan_t > 0.5:
            self.publish_cmd(0.0, 0.0)
            self.get_logger().warn('Sin /scan reciente: robot parado', throttle_duration_sec=2.0)
            return

        pose = self._odom_pose()
        est = self.tracker.get(pose, now)
        pts = self.detector.last_points
        v, w, info = self.controller.step(est, pose, pts, self.docked, now)
        if not self.enabled:
            v, w = 0.0, 0.0
        self.publish_cmd(v, w)
        self.publish_debug(est, info, now, v, w)

        if self.controller.state == 'DOCKED' and not self.finished_logged:
            self.finished_logged = True
            self.get_logger().info(
                f'*** ACOPLADO en {now - self.t_start:.1f} s | lat={info.get("lat", float("nan"))*1000:.1f} mm '
                f'| rumbo={math.degrees(info.get("head", float("nan"))):.2f} deg '
                f'| reintentos={self.controller.retries} ***')

    def publish_cmd(self, v, w):
        self.last_cmd = (v, w)
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def publish_debug(self, est, info, now, v, w):
        det = self.last_det
        st = {
            'state': self.controller.state,
            't': round(now - (self.t_start if self.t_start is not None else now), 2),
            'detected': det is not None,
            'det_rate': round(self.n_dets / max(self.n_scans, 1), 2),
            'v': round(v, 3), 'w': round(w, 3),
            'retries': self.controller.retries,
            'docked': self.docked,
        }
        if 'along' in info:
            st.update(along=round(info['along'], 3), lat_mm=round(info['lat'] * 1000, 1),
                      head_deg=round(math.degrees(info['head']), 2), age=round(info['age'], 2))
        if det is not None:
            st.update(widths_cm=[round(det.widths[0] * 100, 1), round(det.widths[1] * 100, 1)],
                      gap_cm=round(det.gap * 100, 1), sep_cm=round(det.separation * 100, 1),
                      depth_cm=round(det.protrusion * 100, 1), box_pts=det.box_point_count,
                      err_cm=round(det.error * 100, 2))
        self.st_pub.publish(String(data=json.dumps(st)))

        ma = MarkerArray()
        life = Duration(seconds=0.3).to_msg()

        def mk(mid, mtype, color, scale):
            m = Marker()
            m.header.frame_id = self.base_frame
            m.ns, m.id, m.type, m.action = 'dock_lidar', mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
            m.lifetime = life
            return m

        P = lambda xy, z=0.18: Point(x=float(xy[0]), y=float(xy[1]), z=z)

        if det is not None:
            m = mk(1, Marker.LINE_STRIP, (0.2, 0.5, 1.0), 0.02)
            m.points = [P(det.wall_p0 + det.wall_s[0] * det.wall_u),
                        P(det.wall_p0 + det.wall_s[1] * det.wall_u)]
            ma.markers.append(m)
            m = mk(2, Marker.LINE_LIST, (1.0, 0.5, 0.0), 0.03)
            for i in range(2):
                a, b = det.endpoints(i)
                m.points += [P(a), P(b)]
            ma.markers.append(m)
            m = mk(3, Marker.POINTS, (1.0, 1.0, 0.0), 0.02)
            m.points = [P(q) for q in det.box_points]
            ma.markers.append(m)
            m = mk(4, Marker.SPHERE, (1.0, 0.0, 1.0), 0.04)
            m.pose.position = P(det.center)
            ma.markers.append(m)
        if est is not None:
            G, n, _ = est
            m = mk(5, Marker.SPHERE, (0.0, 1.0, 0.0), 0.05)
            m.pose.position = P(G)
            ma.markers.append(m)
            m = mk(6, Marker.LINE_STRIP, (0.0, 1.0, 0.0), 0.01)
            m.points = [P(G), P(G + 1.5 * n)]
            ma.markers.append(m)
        if 'target' in info:
            m = mk(7, Marker.SPHERE, (1.0, 1.0, 1.0), 0.05)
            m.pose.position = P(info['target'])
            ma.markers.append(m)
        m = mk(8, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0), 0.08)
        m.pose.position = Point(x=0.0, y=0.0, z=0.5)
        txt = self.controller.state
        if 'along' in info:
            txt += f'\nd={info["along"]:.2f}m lat={info["lat"]*100:.1f}cm rumbo={math.degrees(info["head"]):.1f}'
        m.text = txt
        ma.markers.append(m)
        self.mk_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = DockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_cmd(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
