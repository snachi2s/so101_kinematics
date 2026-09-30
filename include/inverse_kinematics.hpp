#pragma once

#include "jacobian.hpp"
#include <random>

namespace so101{
    enum class IKMode { Position, Pose };

    struct IKOptions {
        IKMode mode = IKMode::Pose;
        double position_weight = 1.0;
        double orientation_weight = 0.1;     // 1 rad of orientation error costs as much as 0.1 m of position error
        double position_tolerance = 1e-4;    // m
        double orientation_tolerance = 1e-3; // rad
        double initial_damping = 1e-3;
        double min_damping = 1e-6;
        double max_damping = 1e3;
        double max_step = 0.2;               // rad, largest joint change per iteration
        double stall_ratio = 1e-3;           // stop when an accepted step improves cost by less than this fraction
        int max_iterations = 200;
        int max_restarts = 50;
        unsigned int seed = 0;
        std::array<bool, 5> locked{};        // locked joints keep their seed value
    };

    // What the solver hands back. Always check success before using the angles: when it is
    // false, angles is the best attempt found and the two errors say how far off it is.
    struct IKResult {
        JointAngles angles{};
        bool success = false;
        int iterations = 0; // summed over all restarts
        int restarts = 0;   // random restarts used after the given seed
        double position_error = 0.0;
        double orientation_error = 0.0;
    };

    struct PoseError {
        Eigen::Vector3d position;    // world frame
        Eigen::Vector3d orientation; // world frame rotation vector
    };

    inline PoseError poseError(const Sophus::SE3d& target, const Sophus::SE3d& current){
        return {target.translation() - current.translation(),
                (target.so3() * current.so3().inverse()).log()};
    }

    inline Eigen::VectorXd weightedResidual(const PoseError& error, const IKOptions& options){
        if(options.mode == IKMode::Position) return options.position_weight * error.position;

        Eigen::VectorXd r(6);
        r << options.position_weight * error.position, options.orientation_weight * error.orientation;
        return r;
    }

    // The Jacobian that matches weightedResidual: the linear-velocity rows (3x5) in Position
    // mode, or all six rows (6x5) in Pose mode
    inline Eigen::MatrixXd weightedJacobian(const JointAngles& angles, const IKOptions& options){
        const Jacobian J = geometricJacobian(angles);
        if(options.mode == IKMode::Position) return options.position_weight * J.topRows<3>();

        Eigen::MatrixXd Jw(6, 5);
        Jw << options.position_weight * J.topRows<3>(), options.orientation_weight * J.bottomRows<3>();
        return Jw;
    }

    inline Eigen::Matrix<double, 5, 1> dampedLeastSquaresStep(const Eigen::MatrixXd& J, const Eigen::VectorXd& e, double damping){
        const Eigen::Matrix<double, 5, 5> A = J.transpose() * J + damping * damping * Eigen::Matrix<double, 5, 5>::Identity();
        return A.ldlt().solve(J.transpose() * e);
    }

    // True when the pose is close enough to the target. Position and orientation have their
    // own tolerances; in Position mode the orientation is ignored.
    inline bool converged(const PoseError& error, const IKOptions& options){
        const bool position_ok = error.position.norm() < options.position_tolerance;
        const bool orientation_ok = options.mode == IKMode::Position || error.orientation.norm() < options.orientation_tolerance;
        return position_ok && orientation_ok;
    }

    // A single number for how good a finished attempt is, using the same weights as the solver.
    inline double weightedCost(const IKResult& result, const IKOptions& options){
        const double p = options.position_weight * result.position_error;
        const double o = options.mode == IKMode::Position ? 0.0 : options.orientation_weight * result.orientation_error;
        return p * p + o * o;
    }

    // Runs the IK once, starting from one set of joint angles.
    inline IKResult solveIKFromSeed(const Sophus::SE3d& target, const JointAngles& seed, const IKOptions& options){
        JointAngles q = clampToLimits(seed);
        PoseError error = poseError(target, forwardKinematics(q));
        double cost = weightedResidual(error, options).squaredNorm();
        double damping = options.initial_damping;

        int iteration = 0;
        while(iteration < options.max_iterations && !converged(error, options)){
            ++iteration;

            Eigen::MatrixXd J = weightedJacobian(q, options);
            const Eigen::VectorXd r = weightedResidual(error, options);
            Eigen::Matrix<double, 5, 1> dq = dampedLeastSquaresStep(J, r, damping);

            bool blocked = false;
            for(std::size_t i=0; i<q.size(); ++i){
                if(options.locked[i]){
                    J.col(i).setZero();
                    blocked = true;
                    continue;
                }
                const bool at_lower = q[i] <= kJointTable[i].lower_limit + 1e-9 && dq[i] < 0.0;
                const bool at_upper = q[i] >= kJointTable[i].upper_limit - 1e-9 && dq[i] > 0.0;
                if(at_lower || at_upper){
                    J.col(i).setZero(); // freeze joints pushing against their limit
                    blocked = true;
                }
            }
            if(blocked) dq = dampedLeastSquaresStep(J, r, damping);

            const double largest = dq.cwiseAbs().maxCoeff();
            if(largest > options.max_step) dq *= options.max_step / largest;

            JointAngles candidate;
            for(std::size_t i=0; i<q.size(); ++i) candidate[i] = q[i] + dq[i];
            candidate = clampToLimits(candidate);

            const PoseError candidate_error = poseError(target, forwardKinematics(candidate));
            const double candidate_cost = weightedResidual(candidate_error, options).squaredNorm();

            if(candidate_cost < cost){
                const bool stalled = cost - candidate_cost < options.stall_ratio * cost;
                q = candidate;
                error = candidate_error;
                cost = candidate_cost;
                damping = std::max(damping * 0.5, options.min_damping);
                if(stalled) break;
            }
            else{
                damping *= 4.0;
                if(damping > options.max_damping) break;
            }
        }

        IKResult result;
        result.angles = q;
        result.success = converged(error, options);
        result.iterations = iteration;
        result.position_error = error.position.norm();
        result.orientation_error = error.orientation.norm();
        return result;
    }

    // the main IK entry point: finds joint angles that put the tool at the target pose.
    inline IKResult solveIK(const Sophus::SE3d& target, const JointAngles& seed, const IKOptions& options = IKOptions()){
        std::mt19937 rng(options.seed);
        IKResult best = solveIKFromSeed(target, seed, options);
        int total_iterations = best.iterations;
        int restarts = 0;

        while(restarts < options.max_restarts && !best.success){
            ++restarts;
            JointAngles restart_seed = randomJointAngles(rng);
            for(std::size_t i=0; i<seed.size(); ++i){
                if(options.locked[i]) restart_seed[i] = seed[i];
            }
            IKResult attempt = solveIKFromSeed(target, restart_seed, options);
            total_iterations += attempt.iterations;
            if(attempt.success || weightedCost(attempt, options) < weightedCost(best, options)) best = attempt;
        }

        best.iterations = total_iterations;
        best.restarts = restarts;
        return best;
    }
}
