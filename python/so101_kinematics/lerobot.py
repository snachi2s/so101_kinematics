"""Glue between LeRobot's SO-101 follower and the kinematics core.

Calibration comes in two layers:
  1. LeRobot's <id>.json (from lerobot-calibrate): motor ticks -> LeRobot degrees.
  2. Our <id>.urdf.json next to it (from scripts/calibrate_urdf.py): LeRobot degrees -> URDF radians.
     If it is missing, or was measured against a different LeRobot calibration, LeRobot's own
     convention (LeRobot degrees = URDF degrees) is used instead, which is good to a few degrees.

LeRobot itself is only imported inside connect_follower, so the rest of the package works without it.
"""
import json
import sys
from pathlib import Path

import numpy as np

from ._core import JOINT_NAMES, LEROBOT_CONVENTION, HardwareCalibration, from_motor_degrees, to_motor_degrees


def joints_from_lerobot(values, calibration):
    """URDF joint angles (radians) from a LeRobot reading.

    Accepts robot.get_observation() ("shoulder_pan.pos" keys) or bus.sync_read("Present_Position")
    ("shoulder_pan" keys). The gripper and any camera entries are ignored.
    """
    deg = [values[f"{j}.pos"] if f"{j}.pos" in values else values[j] for j in JOINT_NAMES]
    return from_motor_degrees(np.asarray(deg, dtype=float), calibration)


def action_from_joints(q, calibration, gripper=None):
    """LeRobot action dict for robot.send_action() from URDF joint angles (radians).

    gripper is passed through unchanged (LeRobot's 0-100 range) when given.
    """
    deg = to_motor_degrees(np.asarray(q, dtype=float), calibration)
    action = {f"{j}.pos": float(d) for j, d in zip(JOINT_NAMES, deg)}
    if gripper is not None:
        action["gripper.pos"] = float(gripper)
    return action


def urdf_calibration_path(lerobot_calibration_fpath):
    """Where the URDF calibration for a LeRobot calibration file lives: <id>.json -> <id>.urdf.json."""
    path = Path(lerobot_calibration_fpath)
    return path.with_name(path.stem + ".urdf.json")


def _fingerprint(lerobot_calibration):
    return {j: [lerobot_calibration[j].homing_offset, lerobot_calibration[j].range_min, lerobot_calibration[j].range_max]
            for j in JOINT_NAMES}


def save_calibration(path, calibration, lerobot_calibration):
    """Writes a measured URDF calibration, tagged with the LeRobot calibration it belongs to."""
    data = {
        "sign": list(calibration.sign),
        "zero_deg": [round(z, 3) for z in calibration.zero_deg],
        "lerobot_calibration": _fingerprint(lerobot_calibration),
    }
    Path(path).write_text(json.dumps(data, indent=4) + "\n")


def load_calibration(path, lerobot_calibration):
    """Returns (HardwareCalibration, description of where it came from).

    Uses the measured file at path if it exists and was measured against this LeRobot calibration,
    otherwise falls back to LeRobot's convention (LeRobot degrees = URDF degrees).
    """
    path = Path(path)
    if not path.exists():
        return LEROBOT_CONVENTION, f"LeRobot convention (no {path.name}; run scripts/calibrate_urdf.py for sub-mm accuracy)"

    data = json.loads(path.read_text())
    if data.get("lerobot_calibration") != _fingerprint(lerobot_calibration):
        return LEROBOT_CONVENTION, (f"LeRobot convention ({path.name} was measured against a different LeRobot "
                                    "calibration; run scripts/calibrate_urdf.py again)")
    return HardwareCalibration(data["sign"], data["zero_deg"]), f"measured ({path})"


def connect_follower(port, id="follower_arm", calibration_dir=None, **config):
    """Opens an SO-101 follower and loads its calibration. Torque is left as it is.

    Returns (robot, calibration): the SOFollower with its bus connected, and the HardwareCalibration
    to pass to joints_from_lerobot / action_from_joints. Raises RuntimeError if LeRobot's calibration
    file is missing or does not match the motors. Extra keyword arguments go to SOFollowerRobotConfig,
    e.g. max_relative_target=5.0. Call robot.configure() before enabling torque and
    robot.bus.disconnect() when done.
    """
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    calibration_dir = Path(calibration_dir).expanduser() if calibration_dir is not None else None
    robot = SOFollower(SOFollowerRobotConfig(port=port, id=id, calibration_dir=calibration_dir, **config))
    if not robot.calibration:
        raise RuntimeError(f"No LeRobot calibration at {robot.calibration_fpath}. Run lerobot-calibrate first.")

    calibration, source = load_calibration(urdf_calibration_path(robot.calibration_fpath), robot.calibration)
    print(f"LeRobot calibration: {robot.calibration_fpath}\nURDF calibration: {source}", file=sys.stderr)

    robot.bus.connect()
    if not robot.bus.is_calibrated:
        robot.bus.disconnect(disable_torque=False)
        raise RuntimeError(f"Motor calibration does not match {robot.calibration_fpath}. Run lerobot-calibrate first.")
    return robot, calibration
