"""SO-101 forward and inverse kinematics, compiled from the C++ library in include/.

Angles are URDF joint angles in radians (shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll), lengths are metres, poses are 4x4 numpy arrays in the base frame.

    import so101_kinematics as sk

    robot, cal = sk.connect_follower("/dev/ttyACM0")
    q = sk.joints_from_lerobot(robot.get_observation(), cal)  # LeRobot degrees -> radians
    tip = sk.forward_kinematics(q)[:3, 3]                     # metres
    result = sk.solve_position([0.25, 0.0, 0.05], seed=q)
    if result:
        robot.send_action(sk.action_from_joints(result.angles, cal))
"""
import numpy as np

from ._core import (
    JOINT_NAMES,
    LEROBOT_CONVENTION,
    LOWER_LIMITS,
    UPPER_LIMITS,
    HardwareCalibration,
    IKMode,
    IKOptions,
    IKResult,
    clamp_to_limits,
    forward_kinematics,
    from_motor_degrees,
    geometric_jacobian,
    joint_frames,
    numerical_jacobian,
    pose_error,
    solve_ik,
    to_motor_degrees,
    within_limits,
)
from .calibrate import REFERENCE_POSE_DEG, fit_calibration
from .lerobot import (
    action_from_joints,
    connect_follower,
    joints_from_lerobot,
    load_calibration,
    save_calibration,
    urdf_calibration_path,
)

__version__ = "0.1.0"


def solve_position(xyz, seed, lock_wrist_roll=True, **options):
    """IK for a tool-tip position only (xyz in metres, base frame).

    The gripper orientation is left free. wrist_roll is locked at its seed value by default,
    because the tip is only 8 mm off the roll axis and the solver would otherwise spin the
    gripper to make tiny position corrections. Extra keyword arguments go to IKOptions.
    """
    opts = IKOptions(mode=IKMode.POSITION, **options)
    if lock_wrist_roll:
        opts.locked = [False, False, False, False, True]
    target = np.eye(4)
    target[:3, 3] = xyz
    return solve_ik(target, seed, opts)


__all__ = [
    "JOINT_NAMES", "LEROBOT_CONVENTION", "LOWER_LIMITS", "UPPER_LIMITS",
    "HardwareCalibration", "IKMode", "IKOptions", "IKResult",
    "clamp_to_limits", "forward_kinematics", "from_motor_degrees", "geometric_jacobian", "joint_frames",
    "numerical_jacobian", "pose_error", "solve_ik", "solve_position", "to_motor_degrees", "within_limits",
    "action_from_joints", "connect_follower", "joints_from_lerobot", "load_calibration", "save_calibration",
    "urdf_calibration_path", "REFERENCE_POSE_DEG", "fit_calibration",
]
