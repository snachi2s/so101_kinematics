"""Checks the installed so101_kinematics module. Run: python python/tests/test_so101_kinematics.py"""
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import so101_kinematics as sk

passed = failed = 0


def check(ok, name, detail=""):
    global passed, failed
    passed += bool(ok)
    failed += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


def random_angles(rng):
    return rng.uniform(sk.LOWER_LIMITS, sk.UPPER_LIMITS)


rng = np.random.default_rng(7)

print("\n--- FK ---")
T0 = sk.forward_kinematics(np.zeros(5))
check(T0.shape == (4, 4) and T0.dtype == np.float64, "returns a 4x4 float64 array")
check(np.allclose(T0[:3, 3], [0.3913, 0.0, 0.2265], atol=1e-4), "zero pose tip matches the C++ demo", f"{T0[:3, 3]}")
check(np.allclose(sk.forward_kinematics([0, 0, 0, 0, 0]), T0), "accepts a plain list")
joints, tool = sk.joint_frames(np.zeros(5))
check(len(joints) == 5 and np.allclose(tool, T0), "joint_frames tool equals forward_kinematics")

worst = 0.0
for _ in range(200):
    q = random_angles(rng)
    R = sk.forward_kinematics(q)[:3, :3]
    worst = max(worst, np.abs(R.T @ R - np.eye(3)).max(), abs(np.linalg.det(R) - 1.0))
check(worst < 1e-9, "rotations valid on random configs", f"worst defect = {worst:.1e}")

print("\n--- Jacobian ---")
worst = max(np.abs(sk.geometric_jacobian(q) - sk.numerical_jacobian(q)).max()
            for q in (random_angles(rng) for _ in range(200)))
check(worst < 1e-6, "geometric == numerical on random configs", f"worst = {worst:.1e}")
check(sk.geometric_jacobian(np.zeros(5)).shape == (6, 5), "shape is 6x5")

print("\n--- IK ---")
successes = 0
worst_pos = 0.0
all_in_limits = True
for _ in range(200):
    q_true = random_angles(rng)
    target = sk.forward_kinematics(q_true)
    r = sk.solve_ik(target, np.zeros(5))
    successes += r.success
    if r.success:
        worst_pos = max(worst_pos, np.linalg.norm(sk.forward_kinematics(r.angles)[:3, 3] - target[:3, 3]))
    all_in_limits &= sk.within_limits(r.angles)
check(successes >= 198, "full-pose round trip success rate", f"{successes}/200")
check(worst_pos < 1e-4, "FK(IK(target)) matches target", f"worst = {worst_pos * 1000:.3f} mm")
check(all_in_limits, "all solutions within joint limits")

seed = random_angles(rng)
r = sk.solve_position([0.25, 0.0, 0.05], seed)
check(bool(r) and abs(r.angles[4] - seed[4]) < 1e-12, "solve_position hits target with wrist_roll locked",
      f"error = {r.position_error * 1000:.3f} mm")

far = np.eye(4)
far[:3, 3] = [1.0, 0.0, 0.0]
r = sk.solve_ik(far, np.zeros(5), sk.IKOptions(mode=sk.IKMode.POSITION, max_restarts=2))
check(not r and r.position_error > 0.4, "unreachable target fails gracefully", repr(r))

opts = sk.IKOptions(mode=sk.IKMode.POSITION, max_restarts=3, position_tolerance=1e-5)
check(opts.mode == sk.IKMode.POSITION and opts.max_restarts == 3, "IKOptions keyword construction")
check(raises(AttributeError, lambda: sk.IKOptions(not_a_field=1)), "unknown IKOptions field is rejected")
opts.locked = [True, False, False, False, False]
check(opts.locked == [True, False, False, False, False], "locked list assignment sticks")

print("\n--- input validation ---")
bad = np.eye(4)
bad[0, 0] = 2.0
check(raises(ValueError, lambda: sk.solve_ik(bad, np.zeros(5))), "non-rotation target raises ValueError")
check(raises(TypeError, lambda: sk.forward_kinematics([0, 0, 0])), "wrong number of joints raises TypeError")

print("\n--- hardware / LeRobot glue ---")
cal = sk.HardwareCalibration([1, -1, 1, 1, -1], [3.0, -12.5, 7.25, 0.0, -90.0])
q = random_angles(rng)
check(np.allclose(sk.from_motor_degrees(sk.to_motor_degrees(q, cal), cal), q, atol=1e-12), "degrees <-> radians round trip")
check(np.allclose(sk.from_motor_degrees(cal.zero_deg, cal), 0.0, atol=1e-12), "zero_deg maps to URDF zero")
check(np.allclose(sk.from_motor_degrees(np.degrees(q), sk.LEROBOT_CONVENTION), q), "LeRobot convention: degrees are URDF degrees")
check(raises(TypeError, lambda: sk.from_motor_degrees(np.zeros(5))), "calibration argument is required")

action = sk.action_from_joints(q, cal, gripper=12.5)
obs = dict(action, **{"front_camera": None})
check(np.allclose(sk.joints_from_lerobot(obs, cal), q), "action -> observation -> joints round trip")
check(action["gripper.pos"] == 12.5 and len(action) == 6, "gripper passed through")
raw = {j: action[f"{j}.pos"] for j in sk.JOINT_NAMES}
check(np.allclose(sk.joints_from_lerobot(raw, cal), q), "accepts bus.sync_read keys")

print("\n--- calibration files ---")
lerobot_cal = {j: SimpleNamespace(homing_offset=100 * i, range_min=900 + i, range_max=3100 - i)
               for i, j in enumerate(sk.JOINT_NAMES)}
with tempfile.TemporaryDirectory() as tmp:
    path = sk.urdf_calibration_path(Path(tmp) / "my_arm.json")
    check(path.name == "my_arm.urdf.json", "stored next to LeRobot's file", path.name)

    loaded, source = sk.load_calibration(path, lerobot_cal)
    check(list(loaded.zero_deg) == [0.0] * 5 and "convention" in source, "no file -> LeRobot convention", source)

    sk.save_calibration(path, cal, lerobot_cal)
    loaded, source = sk.load_calibration(path, lerobot_cal)
    check(list(loaded.sign) == list(cal.sign) and np.allclose(loaded.zero_deg, cal.zero_deg) and "measured" in source,
          "saved file is loaded back", source)

    lerobot_cal["elbow_flex"].range_min += 10
    loaded, source = sk.load_calibration(path, lerobot_cal)
    check(list(loaded.zero_deg) == [0.0] * 5 and "different LeRobot" in source,
          "file from another LeRobot calibration is ignored", source)

print("\n--- calibration fit ---")
true_cal = sk.HardwareCalibration([1] * 5, [-1.5, -4.6, 4.3, 4.0, 6.7])
reference = sk.to_motor_degrees(np.radians(sk.REFERENCE_POSE_DEG), true_cal)
reference[1] -= 2.5
reference[2] += 2.5
touches = []
while len(touches) < 5:
    r = sk.solve_position([rng.uniform(0.18, 0.38), rng.uniform(-0.2, 0.2), -0.009], random_angles(rng), lock_wrist_roll=False)
    if r:
        touches.append(sk.to_motor_degrees(r.angles, true_cal))
fitted, report = sk.fit_calibration(reference, touches)
err = np.abs(np.array(fitted.zero_deg) - np.array(true_cal.zero_deg)).max()
check(err < 0.05, "recovers a 2.5 deg 'upper arm vertical' error", f"worst zero error = {err:.3f} deg")
check(report["spread_after"] < 1e-4 < report["spread_before"], "table heights flattened",
      f"{report['spread_before'] * 1000:.1f} -> {report['spread_after'] * 1000:.2f} mm")
check(abs(report["table_height"] + 0.009) < 1e-4, "table height recovered", f"{report['table_height'] * 1000:.2f} mm")

print("\n--- speed ---")
q = random_angles(rng)
target = sk.forward_kinematics(q)
n = 2000
t0 = time.perf_counter()
for _ in range(n):
    sk.solve_position(target[:3, 3], q + 0.02)
dt = (time.perf_counter() - t0) / n * 1e6
check(dt < 1000, "warm-start position IK from Python", f"{dt:.1f} us per call")

print(f"\n==== {passed} passed, {failed} failed ====")
sys.exit(1 if failed else 0)
