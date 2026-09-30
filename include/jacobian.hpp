#pragma once

#include "kinematics.hpp"

namespace so101{
    // rows 0-2: linear velocity of the tool point, rows 3-5: angular velocity, both in world frame
    using Jacobian = Eigen::Matrix<double, 6, 5>;

    inline Jacobian numericalJacobian(const JointAngles& angles, double h = 1e-6){
        Jacobian J;

        for(std::size_t i=0; i<angles.size(); ++i){
            JointAngles plus = angles;
            JointAngles minus = angles;
            plus[i] += h;
            minus[i] -= h;

            const Sophus::SE3d T_plus = forwardKinematics(plus);
            const Sophus::SE3d T_minus = forwardKinematics(minus);

            J.col(i).head<3>() = (T_plus.translation() - T_minus.translation()) / (2.0 * h);
            J.col(i).tail<3>() = (T_plus.so3() * T_minus.so3().inverse()).log() / (2.0 * h);
        }
        return J;
    }

    inline Jacobian geometricJacobian(const JointAngles& angles){
        const JointFrames frames = jointFrames(angles);
        const Eigen::Vector3d p_tool = frames.tool.translation();
        Jacobian J;

        for(std::size_t i=0; i<angles.size(); ++i){
            const Eigen::Vector3d z = frames.joints[i].so3().matrix().col(2);
            const Eigen::Vector3d p = frames.joints[i].translation();

            J.col(i).head<3>() = z.cross(p_tool - p);
            J.col(i).tail<3>() = z;
        }
        return J;
    }
}
