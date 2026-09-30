# Streams SO-101 follower joints with torque OFF and prints the FK tip position.
# usage: python scripts/stream_joints.py --port /dev/ttyACM0
#        add --raw to print only the LeRobot degrees (one line per sample, for logging)
import argparse
import sys
import time

import numpy as np
import so101_kinematics as sk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="follower_arm")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--raw", action="store_true", help="print LeRobot degrees only")
    args = parser.parse_args()

    try:
        robot, calibration = sk.connect_follower(args.port, args.id, args.calibration_dir)
    except RuntimeError as e:
        sys.exit(str(e))

    robot.bus.disable_torque()
    print("Torque OFF: move the arm by hand. Ctrl+C to stop.", file=sys.stderr)

    try:
        while True:
            present = robot.bus.sync_read("Present_Position")
            deg = [present[j] for j in sk.JOINT_NAMES]
            if args.raw:
                print(" ".join(f"{d:.2f}" for d in deg), flush=True)
            else:
                q = sk.joints_from_lerobot(present, calibration)
                tip_mm = sk.forward_kinematics(q)[:3, 3] * 1000.0
                warning = "" if sk.within_limits(q) else "  OUTSIDE URDF LIMITS"
                print("motor deg [" + " ".join(f"{d:7.1f}" for d in deg) + " ]"
                      "  urdf deg [" + " ".join(f"{d:7.1f}" for d in np.degrees(q)) + " ]"
                      "  tip mm [" + " ".join(f"{v:7.1f}" for v in tip_mm) + " ]" + warning, flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        robot.bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
