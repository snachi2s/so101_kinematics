# Moves the SO-101 follower tip to (x, y, z) mm in the base frame using so101_kinematics.
# Approaches from above, aborts if a joint lags the plan, returns to the start pose before releasing torque.
# usage: python scripts/move_to.py --port /dev/ttyACM0 --x 250 --y 0 --z 41
import argparse
import logging
import sys
import time

import numpy as np
import so101_kinematics as sk

JOINTS = sk.JOINT_NAMES


def solve_ik(seed_deg, xyz_mm, calibration):
    result = sk.solve_position(np.asarray(xyz_mm) / 1000.0, sk.from_motor_degrees(seed_deg, calibration))
    return list(sk.to_motor_degrees(result.angles, calibration)) if result else None


def read_joints(robot):
    present = robot.bus.sync_read("Present_Position")
    return [present[j] for j in JOINTS]


def send(robot, joints, gripper):
    action = {f"{j}.pos": v for j, v in zip(JOINTS, joints)}
    action["gripper.pos"] = gripper
    robot.send_action(action)


def move(robot, start, goal, gripper, duration, max_lag, tolerance, settle_time, max_correction=8.0):
    steps = max(1, int(duration * 25))
    worst_lag = 0.0
    for k in range(1, steps + 1):
        a = k / steps
        planned = [s + (g - s) * a for s, g in zip(start, goal)]
        send(robot, planned, gripper)
        time.sleep(duration / steps)

        actual = read_joints(robot)
        lags = [abs(p - q) for p, q in zip(planned, actual)]
        worst_lag = max(worst_lag, max(lags))
        if max(lags) > max_lag:
            send(robot, actual, gripper)
            joint = JOINTS[lags.index(max(lags))]
            print(f"  ABORT: {joint} lags the plan by {max(lags):.1f} deg, holding here", flush=True)
            return False

    # the servos' low P gain leaves a steady error under gravity: push the command past the goal until it arrives
    correction = [0.0] * len(goal)
    deadline = time.time() + settle_time
    while time.time() < deadline:
        send(robot, [g + c for g, c in zip(goal, correction)], gripper)
        time.sleep(0.04)
        errors = [g - q for g, q in zip(goal, read_joints(robot))]
        error = max(abs(e) for e in errors)
        if error < tolerance:
            print(f"  reached (worst lag {worst_lag:.1f} deg, final error {error:.1f} deg, "
                  f"max correction {max(abs(c) for c in correction):.1f} deg)", flush=True)
            return True
        correction = [max(-max_correction, min(max_correction, c + 0.1 * e)) for c, e in zip(correction, errors)]

    worst = max(range(len(errors)), key=lambda i: abs(errors[i]))
    print(f"  did NOT settle: {JOINTS[worst]} off by {errors[worst]:.1f} deg (worst lag {worst_lag:.1f} deg)", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="follower_arm")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--approach", type=float, default=60.0, help="mm above the target to pass through first")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds per motion segment")
    parser.add_argument("--hold", type=float, default=5.0, help="seconds to hold at the target")
    parser.add_argument("--max-lag", type=float, default=10.0, help="abort if a joint lags the plan by more (deg)")
    parser.add_argument("--tolerance", type=float, default=1.5, help="deg, per joint, to count as reached")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="plan only, torque stays off")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)

    try:
        robot, calibration = sk.connect_follower(args.port, args.id, args.calibration_dir, max_relative_target=5.0)
    except RuntimeError as e:
        sys.exit(str(e))

    torque_on = False
    try:
        present = robot.bus.sync_read("Present_Position")
        start = [present[j] for j in JOINTS]
        gripper = present["gripper"]

        above = solve_ik(start, (args.x, args.y, args.z + args.approach), calibration)
        target = solve_ik(above, (args.x, args.y, args.z), calibration) if above else None
        if above is None or target is None:
            print("IK failed: target or approach point unreachable")
            return

        print("start  deg:", " ".join(f"{d:7.1f}" for d in start))
        print("above  deg:", " ".join(f"{d:7.1f}" for d in above))
        print("target deg:", " ".join(f"{d:7.1f}" for d in target))
        if args.dry_run:
            return
        if not args.yes and input("Press ENTER to move, anything else to abort: ").strip():
            return

        robot.bus.sync_write("Goal_Position", present)  # hold the current pose when torque turns on
        robot.configure()
        torque_on = True

        seg = dict(max_lag=args.max_lag, tolerance=args.tolerance, settle_time=3.0)
        print("-> above target", flush=True)
        if move(robot, start, above, gripper, args.duration, **seg):
            print("-> down to target", flush=True)
            if move(robot, read_joints(robot), target, gripper, args.duration / 2, **seg):
                print("final  deg:", " ".join(f"{d:7.1f}" for d in read_joints(robot)), flush=True)
                print(f"measure the tip now: target was ({args.x}, {args.y}, {args.z}) mm", flush=True)
                time.sleep(args.hold)
                print("-> back up", flush=True)
                move(robot, read_joints(robot), above, gripper, args.duration / 2, **seg)

        print("-> returning to the start pose", flush=True)
        move(robot, read_joints(robot), start, gripper, args.duration, max_lag=45.0, tolerance=3.0, settle_time=3.0)
        time.sleep(0.5)
    finally:
        robot.bus.disconnect(disable_torque=torque_on)


if __name__ == "__main__":
    main()
