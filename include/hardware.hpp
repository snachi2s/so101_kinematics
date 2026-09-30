#pragma once

#include "kinematics.hpp"
#include <array>

namespace so101{
    using MotorDegrees = std::array<double, 5>; // LeRobot readings (use_degrees=True): shoulder_pan ... wrist_roll

    // How the real arm's joint readings line up with the URDF model: q = sign * (deg - zero_deg).
    // LeRobot puts 0 deg at the middle of each joint's calibrated range, which is close to but not
    // exactly the URDF zero pose, and a motor could count opposite to the URDF joint axis.
    struct HardwareCalibration {
        std::array<double, 5> sign;     // +1 if LeRobot's positive direction matches the URDF axis, else -1
        std::array<double, 5> zero_deg; // LeRobot reading when the arm is in the URDF zero pose
    };

    // LeRobot's own convention: its degrees are URDF degrees. Accurate to a few degrees per joint
    // (about 3 mm height spread at the tip on our arm); a measured calibration, written by
    // scripts/calibrate_urdf.py and loaded in Python, brings that below 1 mm.
    inline const HardwareCalibration kHardwareCalibration = {
        {1.0, 1.0, 1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0, 0.0, 0.0}
    };

    // Converts a reading from the real arm into joint angles the kinematics code understands.
    // Takes the five LeRobot joint positions in degrees (gripper excluded), removes each joint's
    // zero offset, applies its sign and converts to radians.
    // Use this on every reading before calling forwardKinematics or seeding the IK.
    inline JointAngles fromMotorDegrees(const MotorDegrees& deg, const HardwareCalibration& c = kHardwareCalibration){
        JointAngles q;
        for(std::size_t i=0; i<q.size(); ++i){
            q[i] = c.sign[i] * (deg[i] - c.zero_deg[i]) * M_PI / 180.0;
        }
        return q;
    }

    // The reverse of fromMotorDegrees: turns URDF joint angles (radians), for example an IK
    // solution, into the LeRobot degree values to send as goal positions to the motors.
    // fromMotorDegrees(toMotorDegrees(q)) gives back q exactly.
    inline MotorDegrees toMotorDegrees(const JointAngles& q, const HardwareCalibration& c = kHardwareCalibration){
        MotorDegrees deg;
        for(std::size_t i=0; i<q.size(); ++i){
            deg[i] = c.zero_deg[i] + c.sign[i] * q[i] * 180.0 / M_PI;
        }
        return deg;
    }
}
