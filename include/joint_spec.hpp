#pragma once

#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <array>

namespace so101 {

    struct JointSpec {
        std::string name;
        Eigen::Vector3d xyz; // fixed offset from parent joint's frame (meters)
        Eigen::Vector3d rpy; //radians
        double lower_limit;
        double upper_limit;
    };


    inline Sophus::SO3d rpyToSO3(const Eigen::Vector3d& rpy){
        using Sophus::SO3d;
        using Eigen::Vector3d;

        SO3d Rx = SO3d::exp(Vector3d(rpy.x(), 0, 0));
        SO3d Ry = SO3d::exp(Vector3d(0, rpy.y(), 0));
        SO3d Rz = SO3d::exp(Vector3d(0, 0, rpy.z()));
        return Rz * Ry * Rx; // ZYX order

    }

    // returns the SE3 transformation from parent joint to this joint's origin
    inline Sophus::SE3d jointOrigin(const JointSpec& j){
        return Sophus::SE3d(rpyToSO3(j.rpy), j.xyz);
    }

    inline const std::array<JointSpec, 6> kJointTable = {{
    {"shoulder_pan",  {0.0388,  0.0,     0.0624},  {M_PI,       0.0,        -M_PI},     -1.9199, 1.9199},
    {"shoulder_lift", {-0.0304, -0.0183, -0.0542}, {-M_PI/2.0,  -M_PI/2.0,   0.0},       -1.7453, 1.7453},
    {"elbow_flex",    {-0.1126, -0.028,  0.0},     {0.0,        0.0,         M_PI/2.0}, -1.69,   1.69},
    {"wrist_flex",    {-0.1349, 0.0052,  0.0},     {0.0,        0.0,        -M_PI/2.0}, -1.6581, 1.6581},
    {"wrist_roll",    {0.0,     -0.0611, 0.0181},  {M_PI/2.0,   0.0487,      M_PI},     -2.7439, 2.8412},
    {"tool_frame",    {-0.0079, -0.0002, -0.0981}, {0.0,        M_PI,        0.0},        0.0,    0.0}
    }};

}