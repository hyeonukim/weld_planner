#!/usr/bin/env python3
"""
weld_ros_bridge.py  —  ROS 2 subscriber that receives weld-path
                        commands from the weld_planner webapp and
                        forwards them to the Fanuc CRx robot via
                        the existing robot_gui_listener topics.

Run this node alongside robot_gui_listener:

    source /opt/ros/humble/setup.bash
    python3 weld_ros_bridge.py --robot-ip 192.168.2.151

Topics subscribed (from weld_planner server.py):
    /weld/seam_start   geometry_msgs/msg/Point   (metres, camera frame)
    /weld/seam_end     geometry_msgs/msg/Point
    /weld/speed        std_msgs/msg/Float32       (mm/s)
    /weld/procedure    std_msgs/msg/Int32
    /weld/schedule     std_msgs/msg/Int32
    /gui/start_weld    std_msgs/msg/Bool           (armed=true only)

Topics published (to robot_gui_listener / Fanuc):
    /robot/weld_start_point  geometry_msgs/msg/Point  (robot frame, metres)
    /robot/weld_end_point    geometry_msgs/msg/Point
    /robot/motion_speed      std_msgs/msg/Float32
    /weld/procedure          std_msgs/msg/Int32
    /weld/schedule           std_msgs/msg/Int32
    /robot/exec_go           std_msgs/msg/Bool

Coordinate frame transform
--------------------------
Camera frame (output of Metric3D back-projection):
    +X right, +Y down, +Z into the scene

Robot world frame (Fanuc CRx):
    +X forward, +Y left, +Z up

The default transform applies a 90deg rotation around X
(camera Y -> -robot Z, camera Z -> robot Y) PLUS the known
camera-to-table offset set via --cam-offset-z.
Refine with a proper hand-eye calibration (save to hand_eye.npy
and pass --hand-eye-npy).
"""

import argparse, sys, time
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Point
    from std_msgs.msg import Float32, Int32, Bool
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[weld_ros_bridge] WARNING: rclpy not found — running in stub mode.")
    print("  Source /opt/ros/humble/setup.bash then rerun.")


DEFAULT_CAM_OFFSET_Z_M = 0.50    # camera height above table surface (metres)

def _default_cam_T_robot(cam_z_offset: float) -> np.ndarray:
    """Approximate camera-frame to robot-base-frame transform (4x4)."""
    R = np.array([
        [ 1,  0,  0],
        [ 0,  0,  1],
        [ 0, -1,  0],
    ], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3]   = cam_z_offset
    return T


def transform_point(pt_camera: np.ndarray, cam_T_robot: np.ndarray) -> np.ndarray:
    """Apply 4x4 homogeneous transform: camera-frame point -> robot-frame point."""
    ph = np.append(pt_camera, 1.0)
    return (cam_T_robot @ ph)[:3]


class WeldRosBridge(Node):
    def __init__(self, cam_T_robot: np.ndarray):
        super().__init__('weld_ros_bridge')
        self._T = cam_T_robot

        self.create_subscription(Point,   '/weld/seam_start', self._on_start,   10)
        self.create_subscription(Point,   '/weld/seam_end',   self._on_end,     10)
        self.create_subscription(Float32, '/weld/speed',      self._on_speed,   10)
        self.create_subscription(Int32,   '/weld/procedure',  self._on_proc,    10)
        self.create_subscription(Int32,   '/weld/schedule',   self._on_sched,   10)
        self.create_subscription(Bool,    '/gui/start_weld',  self._on_trigger, 10)

        self._pub_start = self.create_publisher(Point,   '/robot/weld_start_point', 10)
        self._pub_end   = self.create_publisher(Point,   '/robot/weld_end_point',   10)
        self._pub_speed = self.create_publisher(Float32, '/robot/motion_speed',      10)
        self._pub_proc  = self.create_publisher(Int32,   '/weld/procedure',          10)
        self._pub_sched = self.create_publisher(Int32,   '/weld/schedule',           10)
        self._pub_go    = self.create_publisher(Bool,    '/robot/exec_go',           10)

        self._start_robot = None
        self._end_robot   = None
        self.get_logger().info('weld_ros_bridge ready — waiting for /weld/seam_start and /weld/seam_end')

    def _xform(self, p: Point) -> Point:
        cam = np.array([p.x, p.y, p.z])
        rob = transform_point(cam, self._T)
        out = Point()
        out.x, out.y, out.z = float(rob[0]), float(rob[1]), float(rob[2])
        return out

    def _on_start(self, msg: Point):
        self._start_robot = self._xform(msg)
        self.get_logger().info(
            f'Seam start (camera): ({msg.x:.4f}, {msg.y:.4f}, {msg.z:.4f}) m  ->  '
            f'(robot): ({self._start_robot.x:.4f}, {self._start_robot.y:.4f}, {self._start_robot.z:.4f}) m')
        self._pub_start.publish(self._start_robot)

    def _on_end(self, msg: Point):
        self._end_robot = self._xform(msg)
        self.get_logger().info(
            f'Seam end   (camera): ({msg.x:.4f}, {msg.y:.4f}, {msg.z:.4f}) m  ->  '
            f'(robot): ({self._end_robot.x:.4f}, {self._end_robot.y:.4f}, {self._end_robot.z:.4f}) m')
        self._pub_end.publish(self._end_robot)

    def _on_speed(self, msg: Float32):
        self._pub_speed.publish(msg)
        self.get_logger().info(f'Speed: {msg.data} mm/s')

    def _on_proc(self, msg: Int32):
        self._pub_proc.publish(msg)

    def _on_sched(self, msg: Int32):
        self._pub_sched.publish(msg)

    def _on_trigger(self, msg: Bool):
        if not msg.data:
            return
        if self._start_robot is None or self._end_robot is None:
            self.get_logger().warn('Trigger received but seam points not yet set — ignoring.')
            return
        go = Bool(); go.data = True
        self._pub_go.publish(go)
        self.get_logger().info('ARMED: exec_go published — robot executing weld path.')


def send_weld_to_robot(
    start_3d_cam: np.ndarray,
    end_3d_cam:   np.ndarray,
    robot_ip:     str   = '192.168.2.151',
    camera_T_robot: np.ndarray = None,
    cam_z_offset: float = DEFAULT_CAM_OFFSET_Z_M,
    procedure:    int   = 1,
    schedule:     int   = 1,
    speed_mm_s:   float = 5.0,
    armed:        bool  = False,
    approach_offset_mm: float = 20.0,
) -> bool:
    """Called from test.py for a one-shot weld command."""
    if not HAS_ROS:
        print('[weld_ros_bridge] Cannot send — rclpy not available.')
        return False

    T = camera_T_robot if camera_T_robot is not None \
        else _default_cam_T_robot(cam_z_offset)

    start_robot = transform_point(start_3d_cam, T)
    end_robot   = transform_point(end_3d_cam,   T)
    print(f'[bridge] Start (robot): {start_robot}')
    print(f'[bridge] End   (robot): {end_robot}')

    rclpy.init(args=None)
    node = rclpy.create_node('weld_bridge_once')
    pub_s  = node.create_publisher(Point,   '/weld/seam_start', 10)
    pub_e  = node.create_publisher(Point,   '/weld/seam_end',   10)
    pub_sp = node.create_publisher(Float32, '/weld/speed',      10)
    pub_p  = node.create_publisher(Int32,   '/weld/procedure',  10)
    pub_sc = node.create_publisher(Int32,   '/weld/schedule',   10)
    pub_go = node.create_publisher(Bool,    '/robot/exec_go',   10)
    time.sleep(0.3)

    def _pt(arr): p = Point(); p.x, p.y, p.z = float(arr[0]), float(arr[1]), float(arr[2]); return p
    pub_s.publish(_pt(start_robot))
    pub_e.publish(_pt(end_robot))
    sp = Float32(); sp.data = float(speed_mm_s); pub_sp.publish(sp)
    pr = Int32();   pr.data = procedure;          pub_p.publish(pr)
    sc = Int32();   sc.data = schedule;           pub_sc.publish(sc)

    if armed:
        go = Bool(); go.data = True; pub_go.publish(go)
        print('[bridge] exec_go published — robot armed.')
    else:
        print('[bridge] Dry-run — exec_go NOT published. Set armed=True to execute.')

    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()
    return True


def main():
    parser = argparse.ArgumentParser(description='Weld-planner to ROS2 bridge node')
    parser.add_argument('--cam-offset-z', type=float, default=DEFAULT_CAM_OFFSET_Z_M,
                        help='Camera height above robot table origin in metres (default 0.5)')
    parser.add_argument('--hand-eye-npy', type=str, default=None,
                        help='Path to 4x4 NumPy .npy hand-eye calibration matrix')
    args = parser.parse_args()

    if not HAS_ROS:
        print('ERROR: rclpy is required.  Run: source /opt/ros/humble/setup.bash')
        sys.exit(1)

    T = np.load(args.hand_eye_npy) if args.hand_eye_npy \
        else _default_cam_T_robot(args.cam_offset_z)

    if args.hand_eye_npy:
        print(f'[bridge] Loaded hand-eye calibration from {args.hand_eye_npy}')
    else:
        print(f'[bridge] Using default transform (cam_z_offset={args.cam_offset_z} m)')
        print('[bridge] For production accuracy, run hand-eye calibration and pass --hand-eye-npy')

    rclpy.init()
    node = WeldRosBridge(T)
    print('[bridge] Spinning — Ctrl+C to stop')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
