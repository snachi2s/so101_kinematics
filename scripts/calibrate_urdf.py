# Measures the LeRobot -> URDF calibration for an SO-101 follower and saves it as <id>.urdf.json
# next to LeRobot's <id>.json. Run lerobot-calibrate first. Torque is OFF: you move the arm by hand.
# usage: python scripts/calibrate_urdf.py --port /dev/ttyACM0 [--id follower_arm]
import argparse
import sys
import time

import numpy as np
import so101_kinematics as sk


def average_reading(robot, seconds=1.0, samples=10):
    readings = []
    for _ in range(samples):
        present = robot.bus.sync_read("Present_Position")
        readings.append([present[j] for j in sk.JOINT_NAMES])
        time.sleep(seconds / samples)
    return np.mean(readings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="follower_arm")
    parser.add_argument("--calibration-dir", default=None)
    args = parser.parse_args()

    try:
        robot, _ = sk.connect_follower(args.port, args.id, args.calibration_dir)
    except RuntimeError as e:
        sys.exit(str(e))
    path = sk.urdf_calibration_path(robot.calibration_fpath)

    try:
        input("\nTorque will be switched OFF. Support the arm, then press ENTER...")
        robot.bus.disable_torque()

        ref = sk.REFERENCE_POSE_DEG
        print("\n1) Reference pose. Put the arm in this pose:\n"
              "   - pointing straight forward\n"
              "   - upper arm vertical (elbow axis directly above the shoulder axis)\n"
              "   - forearm horizontal\n"
              "   - wrist and gripper horizontal, pointing forward, gripper upright\n"
              f"   (URDF angles: pan {ref[0]:.2f}, lift {ref[1]:.2f}, elbow {ref[2]:.2f}, "
              f"wrist_flex {ref[3]:.2f}, roll {ref[4]:.2f} deg)")
        input("   Hold it still and press ENTER...")
        reference = average_reading(robot)
        print("   reading:", " ".join(f"{d:7.2f}" for d in reference))
        first_guess, _ = sk.fit_calibration(reference)

        print("\n2) Table touches. Touch the gripper tip to the table, hold still, press ENTER.\n"
              "   Spread them out: close, far, left, right. At least 3; type d + ENTER when done.")
        touches = []
        while True:
            answer = input(f"   touch {len(touches) + 1}: ").strip().lower()
            if answer == "d":
                if len(touches) >= 3 or input("   fewer than 3 touches: skip refinement? [y/N] ").strip().lower() == "y":
                    break
                continue
            reading = average_reading(robot)
            touches.append(reading)
            tip = sk.forward_kinematics(sk.from_motor_degrees(reading, first_guess))[:3, 3] * 1000.0
            print(f"      tip (reference-pose calibration) = ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}) mm")

        calibration, report = sk.fit_calibration(reference, touches)
        print("\nzero_deg:", " ".join(f"{z:7.2f}" for z in calibration.zero_deg))
        if report["spread_after"] is not None:
            print(f"lift/elbow correction: {report['correction_deg']:+.2f} deg")
            print(f"table height spread: {report['spread_before'] * 1000:.1f} mm -> {report['spread_after'] * 1000:.1f} mm "
                  f"(table at z = {report['table_height'] * 1000:.1f} mm in the base frame)")
            if abs(abs(report["correction_deg"]) - 8.0) < 0.05:
                print("WARNING: correction hit its 8 deg limit; check the reference pose and try again.")

        if input(f"\nSave to {path}? [Y/n] ").strip().lower() in ("", "y", "yes"):
            sk.save_calibration(path, calibration, robot.calibration)
            print("Saved. connect_follower() will use it from now on.")
    finally:
        robot.bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
