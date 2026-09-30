#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include "inverse_kinematics.hpp"
#include "hardware.hpp"

namespace {

int g_passed = 0;
int g_failed = 0;
constexpr double kTol = 1e-9;
constexpr int kNumRandom = 1000;

void check(bool ok, const std::string& name, const std::string& detail = ""){
    if(ok) ++g_passed; else ++g_failed;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name;
    if(!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << "\n";
}

void section(const std::string& title){
    std::cout << "\n--- " << title << " ---\n";
}

std::string sci(double v){
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(3) << v;
    return ss.str();
}

std::string fixed(double v, int precision = 3){
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << v;
    return ss.str();
}

std::string toString(const so101::JointAngles& q){
    std::ostringstream ss;
    ss << std::showpos << std::fixed << std::setprecision(3) << "[" << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << " " << q[4] << "]";
    return ss.str();
}

bool isFinite(const so101::JointAngles& q){
    return std::all_of(q.begin(), q.end(), [](double v){ return std::isfinite(v); });
}

// ---------- independent FK: plain Eigen matrices, no Sophus ----------

Eigen::Matrix3d rawRx(double a){
    const double c = std::cos(a), s = std::sin(a);
    Eigen::Matrix3d R;
    R << 1, 0,  0,
         0, c, -s,
         0, s,  c;
    return R;
}

Eigen::Matrix3d rawRy(double a){
    const double c = std::cos(a), s = std::sin(a);
    Eigen::Matrix3d R;
    R <<  c, 0, s,
          0, 1, 0,
         -s, 0, c;
    return R;
}

Eigen::Matrix3d rawRz(double a){
    const double c = std::cos(a), s = std::sin(a);
    Eigen::Matrix3d R;
    R << c, -s, 0,
         s,  c, 0,
         0,  0, 1;
    return R;
}

Eigen::Matrix4d rawTransform(const Eigen::Matrix3d& R, const Eigen::Vector3d& p){
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = p;
    return T;
}

Eigen::Matrix4d rawOrigin(const so101::JointSpec& j){
    return rawTransform(rawRz(j.rpy.z()) * rawRy(j.rpy.y()) * rawRx(j.rpy.x()), j.xyz);
}

Eigen::Matrix4d rawJointFrame(const so101::JointAngles& q, std::size_t k){
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for(std::size_t i=0; i<=k; ++i){
        T = T * rawOrigin(so101::kJointTable[i]) * rawTransform(rawRz(q[i]), Eigen::Vector3d::Zero());
    }
    return T;
}

Eigen::Matrix4d rawForwardKinematics(const so101::JointAngles& q){
    return rawJointFrame(q, 4) * rawOrigin(so101::kJointTable[5]);
}

// ---------- helpers ----------

double maxAbsDiff(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B){
    return (A - B).cwiseAbs().maxCoeff();
}

double rotationDefect(const Eigen::Matrix3d& R){
    const double orth = (R.transpose() * R - Eigen::Matrix3d::Identity()).norm();
    const double det = std::abs(R.determinant() - 1.0);
    return std::max(orth, det);
}

double reachUpperBound(){
    double sum = 0.0;
    for(const auto& j : so101::kJointTable) sum += j.xyz.norm();
    return sum;
}

std::vector<so101::JointAngles> namedConfigs(){
    std::vector<so101::JointAngles> configs;
    configs.push_back({0, 0, 0, 0, 0});
    so101::JointAngles all_lo{}, all_hi{};
    for(std::size_t i=0; i<5; ++i){
        so101::JointAngles lo{}, hi{};
        lo[i] = all_lo[i] = so101::kJointTable[i].lower_limit;
        hi[i] = all_hi[i] = so101::kJointTable[i].upper_limit;
        configs.push_back(lo);
        configs.push_back(hi);
    }
    configs.push_back(all_lo);
    configs.push_back(all_hi);
    configs.push_back({M_PI / 2.0, 0, 0, 0, 0});
    return configs;
}

so101::JointAngles randomInnerAngles(std::mt19937& rng, double fraction){
    so101::JointAngles q;
    for(std::size_t i=0; i<q.size(); ++i){
        const double mid = 0.5 * (so101::kJointTable[i].lower_limit + so101::kJointTable[i].upper_limit);
        const double half = 0.5 * (so101::kJointTable[i].upper_limit - so101::kJointTable[i].lower_limit) * fraction;
        std::uniform_real_distribution<double> dist(mid - half, mid + half);
        q[i] = dist(rng);
    }
    return q;
}

Sophus::SE3d rotateAboutTip(const Sophus::SE3d& T, const Eigen::Vector3d& world_axis, double angle){
    return Sophus::SE3d(Sophus::SO3d::exp(world_axis.normalized() * angle) * T.so3(), T.translation());
}

const so101::JointAngles kStretched = {0.0, 0.5, -1.288, -0.088, 0.0};

// ---------- 1. forward kinematics ----------

void testSophusVsRaw(){
    section("FK: Sophus vs raw Eigen");
    for(const auto& q : namedConfigs()){
        const Eigen::Matrix4d A = so101::forwardKinematics(q).matrix();
        const Eigen::Matrix4d B = rawForwardKinematics(q);
        check(A.isApprox(B, kTol), "match at " + toString(q), "max |diff| = " + sci(maxAbsDiff(A, B)));
    }

    std::mt19937 rng(42);
    int failures = 0;
    double worst = 0.0;
    for(int k=0; k<kNumRandom; ++k){
        const so101::JointAngles q = so101::randomJointAngles(rng);
        const Eigen::Matrix4d A = so101::forwardKinematics(q).matrix();
        const Eigen::Matrix4d B = rawForwardKinematics(q);
        if(!A.isApprox(B, kTol)) ++failures;
        worst = std::max(worst, maxAbsDiff(A, B));
    }
    check(failures == 0, "match on " + std::to_string(kNumRandom) + " random configs",
          std::to_string(failures) + " failures, worst |diff| = " + sci(worst));
}

void testRotationValidity(){
    section("FK: rotation validity (R^T R = I, det R = 1)");
    std::mt19937 rng(7);
    double worst_sophus = 0.0, worst_raw = 0.0, worst_bottom = 0.0;
    for(int k=0; k<kNumRandom; ++k){
        const so101::JointAngles q = so101::randomJointAngles(rng);
        const Eigen::Matrix4d A = so101::forwardKinematics(q).matrix();
        const Eigen::Matrix4d B = rawForwardKinematics(q);
        worst_sophus = std::max(worst_sophus, rotationDefect(A.block<3, 3>(0, 0)));
        worst_raw = std::max(worst_raw, rotationDefect(B.block<3, 3>(0, 0)));
        worst_bottom = std::max(worst_bottom, (B.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm());
    }
    check(worst_sophus < kTol, "Sophus rotation is valid", "worst defect = " + sci(worst_sophus));
    check(worst_raw < kTol, "raw rotation is valid", "worst defect = " + sci(worst_raw));
    check(worst_bottom < kTol, "raw 4x4 bottom row is [0 0 0 1]", "worst = " + sci(worst_bottom));
}

void testReachBound(){
    section("FK: reach bound (triangle inequality)");
    const double bound = reachUpperBound();
    std::mt19937 rng(123);
    double max_reach = 0.0;
    int violations = 0;
    for(int k=0; k<kNumRandom; ++k){
        const double r = so101::forwardKinematics(so101::randomJointAngles(rng)).translation().norm();
        max_reach = std::max(max_reach, r);
        if(r > bound + kTol) ++violations;
    }
    check(violations == 0, "tip distance <= sum of link offsets",
          "bound = " + fixed(bound, 4) + " m, max seen = " + fixed(max_reach, 4) + " m");
}

void testPanInvariance(){
    section("FK: pan moves tip on a circle about the pan axis");
    const Sophus::SE3d T1 = so101::jointOrigin(so101::kJointTable[0]);
    const Eigen::Vector3d axis_point = T1.translation();
    const Eigen::Vector3d axis_dir = T1.so3().matrix().col(2);

    auto heightAndRadius = [&](const so101::JointAngles& q){
        const Eigen::Vector3d v = so101::forwardKinematics(q).translation() - axis_point;
        const double h = v.dot(axis_dir);
        return Eigen::Vector2d(h, (v - h * axis_dir).norm());
    };

    std::mt19937 rng(99);
    std::uniform_real_distribution<double> pan(so101::kJointTable[0].lower_limit, so101::kJointTable[0].upper_limit);
    double worst = 0.0;
    for(int k=0; k<200; ++k){
        so101::JointAngles q = so101::randomJointAngles(rng);
        const Eigen::Vector2d before = heightAndRadius(q);
        q[0] = pan(rng);
        worst = std::max(worst, (heightAndRadius(q) - before).cwiseAbs().maxCoeff());
    }
    check(worst < kTol, "height and radius about pan axis unchanged", "worst change = " + sci(worst) + " m");
}

// ---------- 2. per-joint frames ----------

void testJointFrames(){
    section("Joint frames");
    std::mt19937 rng(5);
    double worst_tool = 0.0, worst_joint = 0.0, worst_parallel = 0.0, worst_perp = 0.0;
    for(int k=0; k<kNumRandom; ++k){
        const so101::JointAngles q = so101::randomJointAngles(rng);
        const so101::JointFrames frames = so101::jointFrames(q);

        worst_tool = std::max(worst_tool, maxAbsDiff(frames.tool.matrix(), so101::forwardKinematics(q).matrix()));
        for(std::size_t i=0; i<5; ++i){
            worst_joint = std::max(worst_joint, maxAbsDiff(frames.joints[i].matrix(), rawJointFrame(q, i)));
        }

        Eigen::Vector3d z[5];
        for(std::size_t i=0; i<5; ++i) z[i] = frames.joints[i].so3().matrix().col(2);
        worst_parallel = std::max({worst_parallel, z[1].cross(z[2]).norm(), z[1].cross(z[3]).norm()});
        worst_perp = std::max({worst_perp, std::abs(z[0].dot(z[1])), std::abs(z[4].dot(z[1]))});
    }
    check(worst_tool < kTol, "frames.tool equals forwardKinematics", "worst = " + sci(worst_tool));
    check(worst_joint < kTol, "every joint frame matches raw partial chain", "worst = " + sci(worst_joint));
    check(worst_parallel < kTol, "shoulder_lift, elbow_flex, wrist_flex axes parallel", "worst |z1 x zk| = " + sci(worst_parallel));
    check(worst_perp < kTol, "pan axis and wrist_roll axis perpendicular to pitch axes", "worst |dot| = " + sci(worst_perp));
}

// ---------- 3/4. jacobians ----------

void testJacobians(){
    section("Jacobian: geometric vs numerical");
    std::mt19937 rng(11);
    double worst = 0.0;
    for(int k=0; k<kNumRandom; ++k){
        const so101::JointAngles q = so101::randomJointAngles(rng);
        worst = std::max(worst, maxAbsDiff(so101::geometricJacobian(q), so101::numericalJacobian(q)));
    }
    check(worst < 1e-7, "match on " + std::to_string(kNumRandom) + " random configs", "worst |diff| = " + sci(worst));

    for(const auto& q : namedConfigs()){
        const double d = maxAbsDiff(so101::geometricJacobian(q), so101::numericalJacobian(q));
        if(d >= 1e-7) check(false, "match at " + toString(q), "|diff| = " + sci(d));
    }
    check(true, "match at all named configs");

    section("Jacobian: finite-difference step size");
    const std::vector<double> steps = {1e-2, 1e-4, 1e-6, 1e-8, 1e-10};
    std::vector<double> errors(steps.size(), 0.0);
    std::mt19937 rng2(12);
    for(int k=0; k<100; ++k){
        const so101::JointAngles q = so101::randomJointAngles(rng2);
        const so101::Jacobian J = so101::geometricJacobian(q);
        for(std::size_t s=0; s<steps.size(); ++s){
            errors[s] = std::max(errors[s], maxAbsDiff(J, so101::numericalJacobian(q, steps[s])));
        }
    }
    for(std::size_t s=0; s<steps.size(); ++s){
        std::cout << "       h = " << sci(steps[s]) << "  worst error = " << sci(errors[s]) << "\n";
    }
    check(errors[2] < errors[0] && errors[2] < errors[4], "h = 1e-6 beats too large (truncation) and too small (round-off)");

    section("Jacobian: 6x5 has rank 5, one twist direction is unreachable");
    std::mt19937 rng3(13);
    double worst_rank = 1e9, worst_null = 0.0;
    for(int k=0; k<200; ++k){
        const so101::JointAngles q = randomInnerAngles(rng3, 0.8);
        const so101::Jacobian J = so101::geometricJacobian(q);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullU);
        worst_rank = std::min(worst_rank, svd.singularValues()(4));
        const Eigen::VectorXd lost = svd.matrixU().col(5);
        worst_null = std::max(worst_null, (lost.transpose() * J).norm());
    }
    check(worst_rank > 1e-4, "5 non-zero singular values at generic configs", "smallest sigma_5 = " + sci(worst_rank));
    check(worst_null < kTol, "left null vector u6 is orthogonal to every column", "worst |u6^T J| = " + sci(worst_null));
}

// ---------- 5/6. inverse kinematics ----------

void testPoseError(){
    section("IK: pose error");
    std::mt19937 rng(21);
    const Sophus::SE3d T = so101::forwardKinematics(so101::randomJointAngles(rng));
    const so101::PoseError zero = so101::poseError(T, T);
    check(zero.position.norm() < kTol && zero.orientation.norm() < kTol, "error of a pose with itself is zero");

    const Eigen::Vector3d omega(0.1, -0.2, 0.05);
    const Eigen::Vector3d shift(0.01, 0.02, -0.03);
    const Sophus::SE3d target(Sophus::SO3d::exp(omega) * T.so3(), T.translation() + shift);
    const so101::PoseError e = so101::poseError(target, T);
    check((e.position - shift).norm() < kTol && (e.orientation - omega).norm() < kTol,
          "error recovers world-frame shift and rotation vector");
}

void testSeeding(){
    section("IK: seeding from current angles");
    std::mt19937 rng(22);
    const so101::JointAngles q_true = randomInnerAngles(rng, 0.8);
    const Sophus::SE3d target = so101::forwardKinematics(q_true);

    const so101::IKResult exact = so101::solveIK(target, q_true);
    check(exact.success && exact.iterations == 0, "seed = solution converges in 0 iterations",
          std::to_string(exact.iterations) + " iterations");

    so101::JointAngles near = q_true;
    for(auto& v : near) v += 0.05;
    const so101::IKResult tracked = so101::solveIK(target, near);
    double angle_diff = 0.0;
    for(std::size_t i=0; i<5; ++i) angle_diff = std::max(angle_diff, std::abs(tracked.angles[i] - q_true[i]));
    check(tracked.success && tracked.restarts == 0 && angle_diff < 1e-2, "nearby seed converges to the nearby solution",
          std::to_string(tracked.iterations) + " iterations, max |dq| = " + sci(angle_diff));
}

void testRoundTrip(so101::IKMode mode, const std::string& name){
    section("IK round trip + statistics: " + name);
    so101::IKOptions options;
    options.mode = mode;

    std::mt19937 rng(mode == so101::IKMode::Pose ? 31 : 32);
    int successes = 0, pose_mismatch = 0, limit_violations = 0, different_angles = 0;
    long total_iterations = 0, total_restarts = 0;
    double worst_position = 0.0, worst_orientation = 0.0;

    for(int k=0; k<kNumRandom; ++k){
        const so101::JointAngles q_true = so101::randomJointAngles(rng);
        const Sophus::SE3d target = so101::forwardKinematics(q_true);
        const so101::IKResult result = so101::solveIK(target, so101::JointAngles{}, options);

        total_iterations += result.iterations;
        total_restarts += result.restarts;
        if(!so101::withinLimits(result.angles)) ++limit_violations;
        if(!result.success) continue;
        ++successes;

        const so101::PoseError e = so101::poseError(target, so101::forwardKinematics(result.angles));
        worst_position = std::max(worst_position, e.position.norm());
        if(mode == so101::IKMode::Pose) worst_orientation = std::max(worst_orientation, e.orientation.norm());
        if(e.position.norm() >= options.position_tolerance ||
           (mode == so101::IKMode::Pose && e.orientation.norm() >= options.orientation_tolerance)) ++pose_mismatch;

        double diff = 0.0;
        for(std::size_t i=0; i<5; ++i) diff = std::max(diff, std::abs(result.angles[i] - q_true[i]));
        if(diff > 1e-2) ++different_angles;
    }

    const double rate = 100.0 * successes / kNumRandom;
    check(rate >= 99.0, "success rate", fixed(rate, 1) + " %");
    check(pose_mismatch == 0, "FK(IK(target)) matches target",
          "worst pos = " + sci(worst_position) + " m, worst ori = " + sci(worst_orientation) + " rad");
    check(limit_violations == 0, "all solutions within joint limits", std::to_string(limit_violations) + " violations");
    std::cout << "       average iterations = " << fixed(double(total_iterations) / kNumRandom, 1)
              << ", average restarts = " << fixed(double(total_restarts) / kNumRandom, 2) << "\n";
    std::cout << "       solutions whose angles differ from the originals = " << different_angles
              << " (same pose, different joint angles)\n";
}

void testUnreachable(){
    section("IK: unreachable target");
    so101::IKOptions options;
    options.mode = so101::IKMode::Position;
    const Sophus::SE3d target(Sophus::SO3d(), Eigen::Vector3d(1.0, 0.0, 0.2));
    const so101::IKResult result = so101::solveIK(target, so101::JointAngles{}, options);

    const double gap = target.translation().norm() - reachUpperBound();
    check(!result.success, "reports failure", "success = " + std::string(result.success ? "true" : "false"));
    check(isFinite(result.angles) && so101::withinLimits(result.angles), "returns finite angles within limits", toString(result.angles));
    check(result.position_error > gap, "reported error exceeds distance beyond max reach",
          "error = " + fixed(result.position_error, 4) + " m, gap = " + fixed(gap, 4) + " m");
    const double true_error = (target.translation() - so101::forwardKinematics(result.angles).translation()).norm();
    check(std::abs(true_error - result.position_error) < kTol, "reported error matches FK of returned angles");
}

void testSingularity(){
    section("IK: near the stretched-arm singularity");
    const Eigen::MatrixXd Jp = so101::geometricJacobian(kStretched).topRows<3>();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jp, Eigen::ComputeFullU);
    const double sigma_min = svd.singularValues()(2);
    check(sigma_min < 1e-3, "position Jacobian nearly singular when stretched", "sigma_min = " + sci(sigma_min));

    const so101::JointFrames frames = so101::jointFrames(kStretched);
    Eigen::Vector3d outward = svd.matrixU().col(2);
    if(outward.dot(frames.tool.translation() - frames.joints[1].translation()) < 0.0) outward = -outward;

    const Eigen::VectorXd e = 0.05 * outward;
    const double damping = 0.05;
    const double undamped = Jp.completeOrthogonalDecomposition().solve(e).norm();
    const double damped = so101::dampedLeastSquaresStep(Jp, e, damping).norm();
    check(damped <= e.norm() / (2.0 * damping) + kTol, "damped step bounded by |e| / (2 lambda)",
          "|dq| = " + fixed(damped) + " rad, bound = " + fixed(e.norm() / (2.0 * damping)) + " rad");
    check(undamped > 100.0 * damped, "undamped pseudo-inverse step explodes", "|dq| = " + sci(undamped) + " rad");

    so101::IKOptions options;
    options.mode = so101::IKMode::Position;
    options.max_restarts = 0;
    const Sophus::SE3d target(frames.tool.so3(), frames.tool.translation() + e);
    const so101::IKResult result = so101::solveIK(target, kStretched, options);
    check(!result.success && isFinite(result.angles) && so101::withinLimits(result.angles),
          "target 5 cm beyond stretched tip fails gracefully",
          "error = " + fixed(result.position_error, 4) + " m after " + std::to_string(result.iterations) + " iterations");
}

void testLostOrientation(){
    section("IK: the orientation direction a 5-DOF arm cannot reach");
    std::mt19937 rng(41);
    const double angle = 0.15;
    int pitch_ok = 0, roll_ok = 0, lost_ok = 0;
    double lost_position = 0.0, lost_orientation = 0.0;
    const int trials = 100;

    for(int k=0; k<trials; ++k){
        so101::JointAngles q = randomInnerAngles(rng, 0.6);
        q[2] = std::abs(q[2]); // elbow bent away from the stretched pose (-1.288) so the wrist stays reachable
        const so101::JointFrames frames = so101::jointFrames(q);
        const Eigen::Vector3d n = frames.joints[1].so3().matrix().col(2);
        const Eigen::Vector3d a = frames.joints[4].so3().matrix().col(2);

        so101::IKOptions options;
        options.max_restarts = 5;
        const so101::IKResult pitch = so101::solveIK(rotateAboutTip(frames.tool, n, angle), q, options);
        const so101::IKResult roll = so101::solveIK(rotateAboutTip(frames.tool, a, angle), q, options);
        const so101::IKResult lost = so101::solveIK(rotateAboutTip(frames.tool, n.cross(a), angle), q, options);

        if(pitch.success) ++pitch_ok;
        if(roll.success) ++roll_ok;
        if(lost.success) ++lost_ok;
        lost_position += lost.position_error / trials;
        lost_orientation += lost.orientation_error / trials;
    }
    check(pitch_ok >= 0.9 * trials, "rotation about pitch axis n is reachable", std::to_string(pitch_ok) + "/" + std::to_string(trials));
    std::cout << "       rotation about roll axis a reachable: " << roll_ok << "/" << trials
              << " (tool point is 8 mm off the roll axis: rolling also moves the tip)\n";
    check(lost_ok == 0, "rotation about n x a is never reachable", std::to_string(lost_ok) + "/" + std::to_string(trials));
    std::cout << "       n x a residual on average: pos = " << fixed(lost_position * 1000.0, 2)
              << " mm, ori = " << fixed(lost_orientation) << " rad (asked for " << fixed(angle) << " rad)\n";
}

void testWeighting(){
    section("IK: orientation weight trades position against orientation");
    const so101::JointAngles q = {0.3, 0.2, -0.4, 0.5, 0.7};
    const so101::JointFrames frames = so101::jointFrames(q);
    const Eigen::Vector3d n = frames.joints[1].so3().matrix().col(2);
    const Eigen::Vector3d a = frames.joints[4].so3().matrix().col(2);
    const Sophus::SE3d target = rotateAboutTip(frames.tool, n.cross(a), 0.3);

    so101::IKOptions relaxed;
    relaxed.orientation_weight = 0.01;
    relaxed.max_restarts = 0;
    so101::IKOptions strict = relaxed;
    strict.orientation_weight = 1.0;

    const so101::IKResult r = so101::solveIK(target, q, relaxed);
    const so101::IKResult s = so101::solveIK(target, q, strict);
    std::cout << "       weight 0.01: pos = " << fixed(r.position_error * 1000.0, 2) << " mm, ori = " << fixed(r.orientation_error) << " rad\n";
    std::cout << "       weight 1.00: pos = " << fixed(s.position_error * 1000.0, 2) << " mm, ori = " << fixed(s.orientation_error) << " rad\n";
    check(s.orientation_error < r.orientation_error && s.position_error > r.position_error,
          "higher orientation weight -> less orientation error, more position error");
}

void testLockedJoint(){
    section("IK: locked joint keeps its seed value");
    std::mt19937 rng(61);
    so101::IKOptions options;
    options.mode = so101::IKMode::Position;
    options.locked[4] = true;
    int successes = 0;
    double worst_drift = 0.0;
    for(int k=0; k<200; ++k){
        const so101::JointAngles q_true = so101::randomJointAngles(rng);
        so101::JointAngles seed = so101::randomJointAngles(rng);
        seed[4] = q_true[4];
        const so101::IKResult result = so101::solveIK(so101::forwardKinematics(q_true), seed, options);
        if(result.success) ++successes;
        worst_drift = std::max(worst_drift, std::abs(result.angles[4] - seed[4]));
    }
    check(worst_drift < kTol, "wrist_roll never moves", "worst drift = " + sci(worst_drift) + " rad");
    check(successes >= 196, "position IK still succeeds with roll locked", std::to_string(successes) + "/200");
}

// ---------- hardware conversion ----------

void testHardwareConversion(){
    section("Hardware: LeRobot degrees <-> URDF radians");
    const so101::HardwareCalibration c = {{1.0, -1.0, 1.0, -1.0, 1.0}, {3.0, -12.5, 7.25, 0.0, -90.0}};
    std::mt19937 rng(51);
    double worst = 0.0;
    for(int k=0; k<kNumRandom; ++k){
        const so101::JointAngles q = so101::randomJointAngles(rng);
        const so101::JointAngles back = so101::fromMotorDegrees(so101::toMotorDegrees(q, c), c);
        for(std::size_t i=0; i<5; ++i) worst = std::max(worst, std::abs(back[i] - q[i]));
    }
    check(worst < kTol, "fromMotorDegrees(toMotorDegrees(q)) == q", "worst = " + sci(worst));

    const so101::JointAngles q = so101::fromMotorDegrees({3.0, -12.5, 7.25, 0.0, -90.0}, c);
    check(std::all_of(q.begin(), q.end(), [](double v){ return std::abs(v) < kTol; }), "zero_deg reading maps to URDF zero");

    const so101::JointAngles plus = so101::fromMotorDegrees({3.0, -2.5, 7.25, 0.0, -90.0}, c);
    check(std::abs(plus[1] + 10.0 * M_PI / 180.0) < kTol, "sign -1 flips direction", "+10 motor deg -> " + fixed(plus[1] * 180.0 / M_PI, 2) + " urdf deg");
}

}

int main(){
    testSophusVsRaw();
    testRotationValidity();
    testReachBound();
    testPanInvariance();
    testJointFrames();
    testJacobians();
    testPoseError();
    testSeeding();
    testRoundTrip(so101::IKMode::Position, "position only (3x5)");
    testRoundTrip(so101::IKMode::Pose, "full pose (6x5, weighted)");
    testUnreachable();
    testSingularity();
    testLostOrientation();
    testWeighting();
    testLockedJoint();
    testHardwareConversion();

    std::cout << "\n==== " << g_passed << " passed, " << g_failed << " failed ====\n";
    return g_failed == 0 ? 0 : 1;
}
