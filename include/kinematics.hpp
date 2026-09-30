#pragma once

#include "joint_spec.hpp"
#include <array>
#include <random>
#include <algorithm>

namespace so101{
    using JointAngles = std::array<double, 5>; // 5 joints, no tool frame

    struct JointFrames {
        std::array<Sophus::SE3d, 5> joints; // world pose of each joint, z-axis = rotation axis
        Sophus::SE3d tool;
    };

    inline JointFrames jointFrames(const JointAngles& angles){
        JointFrames frames;
        Sophus::SE3d T; // identity transformation

        for(std::size_t i=0; i<angles.size(); ++i){
            Sophus::SE3d origin = jointOrigin(kJointTable[i]);
            Sophus::SE3d rotation(Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, angles[i])), Eigen::Vector3d::Zero());
            T = T * origin * rotation;
            frames.joints[i] = T;
        }

        frames.tool = T * jointOrigin(kJointTable[5]); // fixed tool frame
        return frames;
    }

    inline Sophus::SE3d forwardKinematics(const JointAngles& angles){
        return jointFrames(angles).tool;
    }

    inline bool withinLimits(const JointAngles& angles, double tol = 1e-12){
        for(std::size_t i=0; i<angles.size(); ++i){
            if(angles[i] < kJointTable[i].lower_limit - tol || angles[i] > kJointTable[i].upper_limit + tol) return false;
        }
        return true;
    }

    inline JointAngles clampToLimits(const JointAngles& angles){
        JointAngles clamped;
        for(std::size_t i=0; i<angles.size(); ++i){
            clamped[i] = std::clamp(angles[i], kJointTable[i].lower_limit, kJointTable[i].upper_limit);
        }
        return clamped;
    }

    inline JointAngles randomJointAngles(std::mt19937& rng){
        JointAngles angles;
        for(std::size_t i=0; i<angles.size(); ++i){
            std::uniform_real_distribution<double> dist(kJointTable[i].lower_limit, kJointTable[i].upper_limit);
            angles[i] = dist(rng);
        }
        return angles;
    }
}
