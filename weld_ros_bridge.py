#!/usr/bin/env python3
"""
weld_ros_bridge.py
==================
ROS 2 bridge: takes the 3-D weld start/end coordinates produced by
test.py (VGGT + Metric3D) and commands the Fanuc CRx robot through the
existing welding_robot_ros topic/service interface.

How it fits in the pipeline
---------------------------
  test.py  ──►  WeldReconstructionResult (namedtuple / dict)
                        │
                        ▼
             WeldRobotBridge (this file)
                        │
          ┌─────────────┼──────────────────────┐
          ▼             ▼                      ▼
  /weld/segments   /robot/exec_start      /gui/connect
    /upload          /exec_end              /gui/start_weld
                     /exec_go               etc.

Register & PR layout (from TP source ROS2_JAN_27.LS)
------------------------------------------------------
  R[2]   PROGRAM_SWITCH   — 1 = ROS2.TP, 2 = MULTIPASS.TP
  R[220] REG_SEG_COUNT    — written by /weld/segments/upload (flat[0])
  R[241] same alias in .LS; listener mirrors it
  R[301..306*N] flat segment array, stride 6:
      [StartPR, EndPR, Proc, Sched, Mode(0=dry/1=weld), Speed_mm_s]
  R[243] REG_WP_START     — written by /robot/exec_start
  R[244] REG_WP_END       — written by /robot/exec_end
  R[245] REG_WP_EXEC      — pulsed True by /robot/exec_go
  R[248] REG_WELD_ARM     — set by /gui/start_weld (armed flag)
  R[247] REG_SPEED_OVR    — set by /gui/speed_override
  PR[N]  position registers — saved via /gui/waypoint/save_pr

Usage
-----
  # Standalone (no GUI needed):
  ros2 run <your_pkg> weld_ros_bridge \
      --robot-ip 192.168.2.151 \
      --start-pr 10 \
      --end-pr   11 \
      --procedure 1 \
      --schedule  1 \
      --speed     5.0 \
      --dry-run         # omit to do a real weld

  # From Python (called at the end of test.py's main()):
  from weld_ros_bridge import send_weld_to_robot
  send_weld_to_robot(start_3d, end_3d, robot_ip="192.168.2.151")
"""

import time
import argparse
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import Bool, Float32, Int32, Int32MultiArray, Empty, String
from geometry_msgs.msg import Vector3
from std_srvs.srv import Trigger


# ─────────────────────────────────────────────────────────────────────────────
# Constants matching welding_robot_ros register map
# ─────────────────────────────────────────────────────────────────────────────

# Position registers used by this bridge.  The TP program reads
# PR[START_PR] as the approach / arc-start position and PR[END_PR] as
# the arc-end / retract position.
DEFAULT_START_PR = 10   # PR index for weld start point
DEFAULT_END_PR   = 11   # PR index for weld end point

# Segment flat array stride (from on_segments_upload in weld_gui.py):
#   [count, startPR, endPR, proc, sched, mode, speed, ...]
SEG_STRIDE = 6

# Robot program IDs (R[2] PROGRAM_SWITCH)
PROG_IDLE     = 0
PROG_ROS2     = 1   # ROS2.TP  — waypoint / segment mode
PROG_MULTIPASS = 2

# Speed limits from TP (MAIN.LS lines 3-6)
SPEED_MIN_MM_S = 1
SPEED_MAX_MM_S = 250


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def metric3d_to_robot_frame(point_m: np.ndarray,
                             camera_T_robot: np.ndarray | None = None
                             ) -> np.ndarray:
    """
    Convert a point from the Metric3D/VGGT camera frame (X right, Y down, Z fwd,
    units = metres) to the Fanuc robot world frame (X fwd, Y left, Z up,
    units = mm).

    Parameters
    ----------
    point_m        : (3,) float, camera-frame XYZ in metres
    camera_T_robot : (4,4) homogeneous transform from camera frame to robot
                     world frame.  If None, a simple axis-swap + scale is used
                     (good enough for bench testing; replace with your
                     hand-eye calibration result for production).

    Returns
    -------
    (3,) float, robot world-frame XYZ in millimetres
    """
    if camera_T_robot is not None:
        p_h = np.array([*point_m, 1.0])          # homogeneous
        p_robot_m = (camera_T_robot @ p_h)[:3]
        return p_robot_m * 1000.0                 # → mm

    # Default: camera Z → robot X, camera X → robot -Y, camera Y → robot -Z
    # (axis swap only, no rotation/translation calibration applied)
    x_cam, y_cam, z_cam = point_m
    x_rob =  z_cam * 1000.0   # depth  → forward
    y_rob = -x_cam * 1000.0   # right  → left
    z_rob = -y_cam * 1000.0   # down   → up
    return np.array([x_rob, y_rob, z_rob], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 bridge node
# ─────────────────────────────────────────────────────────────────────────────

class WeldRobotBridge(Node):
    """
    Thin ROS 2 node that:
      1. Publishes /gui/connect to bring the listener online.
      2. Saves the weld start & end positions into robot PR registers by
         publishing /gui/waypoint/save_pr (the TP program captures LPOS
         into PR[N] when R[246]=N).
      3. Uploads a single weld segment via /weld/segments/upload.
      4. Sets program to ROS2.TP and fires /robot/exec_go to run the weld.
      5. Monitors /gui/status for completion / fault feedback.
    """

    def __init__(self,
                 start_xyz_mm: np.ndarray,
                 end_xyz_mm: np.ndarray,
                 start_pr: int   = DEFAULT_START_PR,
                 end_pr: int     = DEFAULT_END_PR,
                 procedure: int  = 1,
                 schedule: int   = 1,
                 speed_mm_s: float = 5.0,
                 armed: bool     = False,
                 approach_offset_mm: float = 20.0):
        """
        Parameters
        ----------
        start_xyz_mm      : weld arc-start, robot world frame, mm
        end_xyz_mm        : weld arc-end,   robot world frame, mm
        start_pr          : PR register index to store start point
        end_pr            : PR register index to store end point
        procedure         : weld procedure number (passed to Weld Start[proc,sched])
        schedule          : weld schedule number
        speed_mm_s        : cartesian travel speed in mm/s (clamped 1–250)
        armed             : if False → dry-run (no arc); if True → live weld
        approach_offset_mm: Z-offset added to start point for the approach PR
        """
        super().__init__('weld_robot_bridge')

        self._start_xyz  = np.asarray(start_xyz_mm, dtype=float)
        self._end_xyz    = np.asarray(end_xyz_mm,   dtype=float)
        self._start_pr   = int(start_pr)
        self._end_pr     = int(end_pr)
        self._procedure  = int(procedure)
        self._schedule   = int(schedule)
        self._speed      = float(np.clip(speed_mm_s, SPEED_MIN_MM_S, SPEED_MAX_MM_S))
        self._armed      = bool(armed)
        self._approach_z = float(approach_offset_mm)

        self._status_log: list[str] = []
        self._done       = threading.Event()
        self._fault      = False

        # ── Publishers (matching weld_gui.py GuiPublisher) ──
        self._pub_connect   = self.create_publisher(Bool,           '/gui/connect',            10)
        self._pub_start_weld= self.create_publisher(Bool,           '/gui/start_weld',         10)
        self._pub_stop_weld = self.create_publisher(Bool,           '/gui/stop_weld',          10)
        self._pub_speed_ovr = self.create_publisher(Float32,        '/gui/speed_override',     10)
        self._pub_arm       = self.create_publisher(Bool,           '/gui/start_weld',         10)
        self._pub_go_home   = self.create_publisher(Bool,           '/gui/go_home',            10)

        self._pub_wp_save   = self.create_publisher(Int32,          '/gui/waypoint/save_pr',   10)
        self._pub_jog_nudge = self.create_publisher(Vector3,        '/gui/jog_nudge',          10)

        self._pub_seg_upload= self.create_publisher(Int32MultiArray,'/weld/segments/upload',   10)
        self._pub_seg_clear = self.create_publisher(Empty,          '/weld/segments/clear',    10)

        self._pub_exec_start= self.create_publisher(Int32,          '/robot/exec_start',       10)
        self._pub_exec_end  = self.create_publisher(Int32,          '/robot/exec_end',         10)
        self._pub_exec_go   = self.create_publisher(Bool,           '/robot/exec_go',          10)
        self._pub_prog      = self.create_publisher(Int32,          '/robot/program_switch',   10)
        self._pub_mot_speed = self.create_publisher(Float32,        '/robot/motion_speed',     10)

        # ── Subscriptions ──
        self.create_subscription(String, '/gui/status', self._on_status, 10)

        self.get_logger().info(
            f"WeldRobotBridge ready │ start={self._start_xyz} mm │ "
            f"end={self._end_xyz} mm │ armed={self._armed}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pub_bool(self, pub, value: bool):
        msg = Bool(); msg.data = value; pub.publish(msg)

    def _pub_int(self, pub, value: int):
        msg = Int32(); msg.data = int(value); pub.publish(msg)

    def _pub_float(self, pub, value: float):
        msg = Float32(); msg.data = float(value); pub.publish(msg)

    def _sleep(self, secs: float):
        """Spin + sleep so subscriptions are processed while waiting."""
        deadline = self.get_clock().now().nanoseconds + int(secs * 1e9)
        while self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _on_status(self, msg: String):
        text = msg.data
        self._status_log.append(text)
        self.get_logger().info(f"[robot status] {text}")
        lower = text.lower()
        if "fault" in lower or "error" in lower or "e-stop" in lower:
            self._fault = True
            self._done.set()
        if "done" in lower or "complete" in lower or "finished" in lower:
            self._done.set()

    def _move_to_xyz(self, xyz_mm: np.ndarray, pr_index: int, label: str):
        """
        Jog the robot to a Cartesian position (in mm, robot world frame)
        and save it as PR[pr_index].

        Strategy: publish a jog nudge that encodes the absolute target
        as a delta from current position — the listener applies it as
        an incremental Cartesian move.  For the initial PR-save workflow
        the robot must already be near the workpiece; this just fine-tunes
        the position.

        For full autonomous positioning (robot at arbitrary start pose)
        you would use MoveIt2 Cartesian planning instead — see
        experiments/scripts/taskspace_plan.py for that pattern.
        """
        self.get_logger().info(
            f"  → Positioning to {label} PR[{pr_index}]: {xyz_mm} mm"
        )
        # 1. Publish the target position as an exec_start/end pair pointing
        #    to a single waypoint PR that the operator has pre-taught at
        #    a safe intermediate pose (we re-use pr_index itself for that).
        self._pub_int(self._pub_exec_start, pr_index)
        self._pub_int(self._pub_exec_end,   pr_index)
        self._sleep(0.1)
        # 2. Trigger the TP to move to that PR
        self._pub_bool(self._pub_exec_go, True)
        self._sleep(0.5)
        # 3. Nudge to exact weld coordinate using /gui/jog_nudge (mm)
        #    The robot listener converts this into a DPM delta move.
        nudge = Vector3()
        nudge.x, nudge.y, nudge.z = float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2])
        self._pub_jog_nudge.publish(nudge)
        self._sleep(0.3)
        # 4. Capture current position into PR[pr_index] via R[246]
        self._pub_int(self._pub_wp_save, pr_index)
        self._sleep(0.3)
        self.get_logger().info(f"  ✓ Saved PR[{pr_index}]")

    # ── Public API ────────────────────────────────────────────────────────────

    def execute_weld(self) -> bool:
        """
        Full weld execution sequence:
          1. Connect & configure
          2. Move to start, save PR; move to end, save PR
          3. Upload segment, fire exec_go
          4. Wait for completion
        Returns True on success, False on fault/timeout.
        """
        log = self.get_logger()

        # ── Step 1: Connect ──────────────────────────────────────────────────
        log.info("Step 1/5 │ Connecting to robot controller...")
        self._pub_bool(self._pub_connect, True)
        self._sleep(1.0)

        # Set speed override (uses 1-100 % scale on /gui/speed_override)
        speed_pct = min(100.0, max(1.0, self._speed / SPEED_MAX_MM_S * 100.0))
        self._pub_float(self._pub_speed_ovr, speed_pct)
        # Also set motion speed in mm/s directly
        self._pub_float(self._pub_mot_speed, self._speed)
        self._sleep(0.2)

        # Arm / disarm
        self._pub_bool(self._pub_arm, self._armed)
        self._sleep(0.2)

        # Select ROS2.TP program (R[2] = 1)
        self._pub_int(self._pub_prog, PROG_ROS2)
        self._sleep(0.2)

        # ── Step 2: Clear any stale segments ────────────────────────────────
        log.info("Step 2/5 │ Clearing previous segments...")
        self._pub_seg_clear.publish(Empty())
        self._sleep(0.3)

        # ── Step 3: Position robot & save PRs ───────────────────────────────
        log.info("Step 3/5 │ Teaching weld start position...")
        # Approach point: lift above start by approach_offset_mm in Z
        approach_xyz = self._start_xyz.copy()
        approach_xyz[2] += self._approach_z
        self._move_to_xyz(approach_xyz,   self._start_pr,     "approach/start")

        log.info("Step 3/5 │ Teaching weld end position...")
        self._move_to_xyz(self._end_xyz,  self._end_pr,       "end")

        # ── Step 4: Upload weld segment ──────────────────────────────────────
        log.info("Step 4/5 │ Uploading weld segment to robot...")
        mode = 1 if self._armed else 0          # 1 = live weld, 0 = dry run
        speed_int = int(round(self._speed))
        flat = [
            1,                   # count = 1 segment
            self._start_pr,      # StartPR
            self._end_pr,        # EndPR
            self._procedure,     # Proc
            self._schedule,      # Sched
            mode,                # Mode
            speed_int,           # Speed mm/s
        ]
        seg_msg = Int32MultiArray(); seg_msg.data = flat
        self._pub_seg_upload.publish(seg_msg)
        self._sleep(0.5)

        log.info(
            f"  Segment: PR[{self._start_pr}]→PR[{self._end_pr}] │ "
            f"proc={self._procedure} sched={self._schedule} │ "
            f"{'WELD' if self._armed else 'DRY-RUN'} @ {speed_int} mm/s"
        )

        # ── Step 5: Execute ──────────────────────────────────────────────────
        log.info("Step 5/5 │ Firing exec_go — robot moving to weld...")
        self._pub_int(self._pub_exec_start, self._start_pr)
        self._pub_int(self._pub_exec_end,   self._end_pr)
        self._sleep(0.15)
        self._pub_bool(self._pub_exec_go, True)

        # Wait for done / fault (30 s timeout)
        timeout_s = 30.0
        finished  = self._done.wait(timeout=timeout_s)

        if not finished:
            log.warning("Timeout waiting for weld completion — stopping robot.")
            self._pub_bool(self._pub_stop_weld, True)
            return False

        if self._fault:
            log.error("Fault reported by robot. Check E-stop / status.")
            return False

        log.info("✓ Weld sequence completed successfully.")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — called from test.py
# ─────────────────────────────────────────────────────────────────────────────

def send_weld_to_robot(
        start_3d_cam: np.ndarray,
        end_3d_cam:   np.ndarray,
        robot_ip:     str   = "192.168.2.151",   # not used directly here;
        # set via ros2 launch / environment before calling
        camera_T_robot: np.ndarray | None = None,
        start_pr:     int   = DEFAULT_START_PR,
        end_pr:       int   = DEFAULT_END_PR,
        procedure:    int   = 1,
        schedule:     int   = 1,
        speed_mm_s:   float = 5.0,
        armed:        bool  = False,
        approach_offset_mm: float = 20.0,
) -> bool:
    """
    Convenience function to call from test.py after weld endpoint detection.

    Parameters
    ----------
    start_3d_cam   : (3,) weld start in Metric3D camera frame, metres
    end_3d_cam     : (3,) weld end   in Metric3D camera frame, metres
    robot_ip       : Fanuc controller IP (used in launch, not this node)
    camera_T_robot : 4×4 hand-eye transform; None = axis-swap default
    armed          : False = dry-run, True = live weld

    Returns True on success.

    Example (from test.py)
    ----------------------
    from weld_ros_bridge import send_weld_to_robot
    success = send_weld_to_robot(avg_start, avg_end, armed=False)
    """
    # Convert from camera frame (metres) to robot frame (mm)
    start_mm = metric3d_to_robot_frame(start_3d_cam, camera_T_robot)
    end_mm   = metric3d_to_robot_frame(end_3d_cam,   camera_T_robot)

    print(f"\n[bridge] Weld start (robot frame): {start_mm} mm")
    print(f"[bridge] Weld end   (robot frame): {end_mm} mm")
    print(f"[bridge] Seam length: {np.linalg.norm(end_mm - start_mm):.1f} mm")

    rclpy.init()
    bridge = WeldRobotBridge(
        start_xyz_mm        = start_mm,
        end_xyz_mm          = end_mm,
        start_pr            = start_pr,
        end_pr              = end_pr,
        procedure           = procedure,
        schedule            = schedule,
        speed_mm_s          = speed_mm_s,
        armed               = armed,
        approach_offset_mm  = approach_offset_mm,
    )

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(bridge)

    # Run execute_weld in a background thread so the executor can spin
    result_holder = [False]
    def _run():
        result_holder[0] = bridge.execute_weld()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    while t.is_alive():
        executor.spin_once(timeout_sec=0.05)
    t.join()

    bridge.destroy_node()
    rclpy.shutdown()
    return result_holder[0]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Send a weld path to the Fanuc CRx robot via ROS 2."
    )
    ap.add_argument("--start",       nargs=3, type=float, required=True,
                    metavar=("X","Y","Z"),
                    help="Weld start in ROBOT frame, mm  (e.g. 400 0 200)")
    ap.add_argument("--end",         nargs=3, type=float, required=True,
                    metavar=("X","Y","Z"),
                    help="Weld end   in ROBOT frame, mm")
    ap.add_argument("--start-pr",    type=int,   default=DEFAULT_START_PR)
    ap.add_argument("--end-pr",      type=int,   default=DEFAULT_END_PR)
    ap.add_argument("--procedure",   type=int,   default=1)
    ap.add_argument("--schedule",    type=int,   default=1)
    ap.add_argument("--speed",       type=float, default=5.0,
                    help="Travel speed mm/s (1–250)")
    ap.add_argument("--approach-offset", type=float, default=20.0,
                    help="Z offset (mm) added to start point for approach")
    ap.add_argument("--armed",       action="store_true",
                    help="Enable live welding (default: dry-run)")
    args = ap.parse_args()

    rclpy.init()
    bridge = WeldRobotBridge(
        start_xyz_mm        = np.array(args.start),
        end_xyz_mm          = np.array(args.end),
        start_pr            = args.start_pr,
        end_pr              = args.end_pr,
        procedure           = args.procedure,
        schedule            = args.schedule,
        speed_mm_s          = args.speed,
        armed               = args.armed,
        approach_offset_mm  = args.approach_offset,
    )

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(bridge)
    result_holder = [False]

    def _run():
        result_holder[0] = bridge.execute_weld()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    while t.is_alive():
        executor.spin_once(timeout_sec=0.05)
    t.join()

    bridge.destroy_node()
    rclpy.shutdown()
    return 0 if result_holder[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
