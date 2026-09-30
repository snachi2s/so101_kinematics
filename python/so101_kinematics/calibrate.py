"""Measuring the LeRobot -> URDF calibration (zero_deg) from a reference pose and table touches."""
import numpy as np

from ._core import HardwareCalibration, forward_kinematics, from_motor_degrees

# URDF angles (deg) of the reference pose: arm pointing straight forward, upper arm vertical
# (elbow axis above the shoulder axis), forearm horizontal, wrist-to-tip horizontal, gripper upright.
REFERENCE_POSE_DEG = np.array([0.0, -13.96, 16.17, -5.05, 0.0])


def zero_from_reference(reference_deg, sign=(1, 1, 1, 1, 1)):
    """zero_deg such that the LeRobot reading taken in the reference pose maps to REFERENCE_POSE_DEG."""
    sign = np.asarray(sign, dtype=float)
    return np.asarray(reference_deg, dtype=float) - sign * REFERENCE_POSE_DEG


def table_heights(touches_deg, calibration):
    """FK tip height (m) for each LeRobot reading taken with the tip touching the table."""
    return np.array([forward_kinematics(from_motor_degrees(t, calibration))[2, 3] for t in touches_deg])


def fit_calibration(reference_deg, touches_deg=(), sign=(1, 1, 1, 1, 1), max_correction_deg=8.0):
    """HardwareCalibration from a reference-pose reading, refined by table touches.

    Judging "upper arm vertical" by eye mostly errs by the same angle on shoulder_lift and,
    oppositely, elbow_flex (the forearm is still levelled). That error tilts the table as seen by
    FK, so with 3 or more touches spread in reach, the lift/elbow correction that makes all touch
    heights equal is found by a 1-D search.

    Returns (calibration, report) where report has the height spread before/after (m), the table
    height in the base frame (m) and the correction applied (deg).
    """
    zero = zero_from_reference(reference_deg, sign)
    touches = np.asarray(touches_deg, dtype=float).reshape(-1, 5)

    def calibration_for(d):
        z = zero.copy()
        z[1] += d
        z[2] -= d
        return HardwareCalibration(list(sign), list(z))

    report = {"correction_deg": 0.0, "spread_before": None, "spread_after": None, "table_height": None}
    if len(touches) < 3:
        return calibration_for(0.0), report

    spread = lambda d: table_heights(touches, calibration_for(d)).std()
    candidates = np.arange(-max_correction_deg, max_correction_deg + 1e-9, 0.01)
    best = min(candidates, key=spread)
    calibration = calibration_for(best)
    report.update(correction_deg=float(best), spread_before=float(spread(0.0)), spread_after=float(spread(best)),
                  table_height=float(table_heights(touches, calibration).mean()))
    return calibration, report
