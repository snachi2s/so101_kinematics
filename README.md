# SO-101 Kinematics

Forward and inverse kinematics for the [LeRobot](https://github.com/huggingface/lerobot) SO-101 arm. The core is written from scratch in C++ (Eigen + Sophus) and you use it from Python with a single import.

## 1. Install

You need a LeRobot conda environment (tested with LeRobot 0.6) and a C++ compiler (`sudo apt install build-essential`). Eigen and Sophus are downloaded automatically if you don't have them.

```bash
conda activate lerobot
git clone <this repo> so101_kinematics
cd so101_kinematics
pip install .
python python/tests/test_so101_kinematics.py   # should end with "0 failed"
```

## 2. Set up your arm

Run these from the repo folder. Find your port with `lerobot-find-port`. Use the same `--id` everywhere; it names your calibration files.

**a. LeRobot calibration** (skip if you've already done it for this arm):

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm
```

This alone is enough to use the library, with the tip accurate to a few millimetres.

**b. Fine calibration** (optional, about 2 minutes, for sub-millimetre accuracy):

```bash
python scripts/calibrate_urdf.py --port /dev/ttyACM0 --id follower_arm
```

Torque turns off, so hold the arm. The script then asks you to:
1. Put the arm in the reference pose: pointing straight forward, upper arm vertical, forearm horizontal, gripper horizontal and upright. A phone level app helps.
2. Touch the gripper tip to the table at 3 or more spots (near, far, left, right), pressing ENTER at each one. Type `d` when you're done.

It saves `follower_arm.urdf.json` next to LeRobot's `follower_arm.json`. If you ever re-run `lerobot-calibrate`, run this again; until then the old file is ignored with a warning.

**c. Check it.** Move the arm by hand and watch the tip position:

```bash
python scripts/stream_joints.py --port /dev/ttyACM0 --id follower_arm
```

Touching the table should give roughly the same `z` everywhere.

**d. First move.** The script shows the plan and waits for ENTER before moving:

```bash
python scripts/move_to.py --port /dev/ttyACM0 --id follower_arm --x 250 --y 0 --z 50
```

`x`, `y`, `z` are in mm in the robot base frame: `x` forward, `y` to the left, `z` up. The tip goes there from above, holds for 5 s, and returns to where it started. Use `--dry-run` to see the plan without moving.

## 3. Use it in your code

```python
import so101_kinematics as sk

robot, cal = sk.connect_follower("/dev/ttyACM0", id="follower_arm")
q = sk.joints_from_lerobot(robot.get_observation(), cal)   # joint angles, radians

tip = sk.forward_kinematics(q)[:3, 3]                     # metres
result = sk.solve_position([0.25, 0.0, 0.05], seed=q)
if result:
    robot.bus.sync_write("Goal_Position", robot.bus.sync_read("Present_Position"))  # hold still...
    robot.configure()                                                               # ...when torque turns on
    robot.send_action(sk.action_from_joints(result.angles, cal))                    # jumps straight there: keep targets close
```

Also available: `solve_ik` (full pose with `IKOptions`), `joint_frames`, `geometric_jacobian`, `numerical_jacobian`. Each has a docstring.

The arm has 5 joints, so it can't reach every position *and* orientation. `solve_position` ignores orientation. `solve_ik` lets you choose how much orientation matters with `orientation_weight`.

## C++

The library is header-only in `include/`. To build and run the C++ tests:

```bash
cmake -S . -B build && cmake --build build && ./build/tests
```
