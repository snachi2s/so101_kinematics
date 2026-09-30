#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include "hardware.hpp"
#include "inverse_kinematics.hpp"

namespace py = pybind11;

// Sophus calls this instead of std::abort() (SOPHUS_ENABLE_ENSURE_HANDLER), so a bad input
// raises a Python RuntimeError rather than killing the interpreter.
namespace Sophus{
    void ensureFailed(const char* function, const char* file, int line, const char* description){
        throw std::runtime_error(std::string("Sophus check failed: ") + description + " (" + file + ":" + std::to_string(line) + ", " + function + ")");
    }
}

namespace{
    using Vector5d = Eigen::Matrix<double, 5, 1>;

    so101::JointAngles toAngles(const Vector5d& v){
        so101::JointAngles q;
        for(std::size_t i=0; i<q.size(); ++i) q[i] = v[i];
        return q;
    }

    Vector5d toVector(const std::array<double, 5>& q){
        Vector5d v;
        for(std::size_t i=0; i<q.size(); ++i) v[i] = q[i];
        return v;
    }

    // Accepts any 4x4 homogeneous transform whose rotation block is a rotation to within 1e-6,
    // then re-normalizes it so small numerical drift from Python-side math is tolerated.
    Sophus::SE3d toSE3(const Eigen::Matrix4d& T){
        const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        if(!(R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-6) || R.determinant() <= 0.0){
            throw py::value_error("target rotation block is not a rotation matrix (need R^T R = I, det R = +1)");
        }
        if(!T.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-9)){
            throw py::value_error("target bottom row must be [0, 0, 0, 1]");
        }
        return Sophus::SE3d(Sophus::SO3d(Eigen::Quaterniond(R).normalized()), T.block<3, 1>(0, 3));
    }
}

PYBIND11_MODULE(_core, m){
    m.doc() = "SO-101 forward/inverse kinematics (C++ core). Angles in radians, lengths in metres.";

    std::vector<std::string> names;
    Vector5d lower, upper;
    for(std::size_t i=0; i<5; ++i){
        names.push_back(so101::kJointTable[i].name);
        lower[i] = so101::kJointTable[i].lower_limit;
        upper[i] = so101::kJointTable[i].upper_limit;
    }
    m.attr("JOINT_NAMES") = names;
    m.attr("LOWER_LIMITS") = lower;
    m.attr("UPPER_LIMITS") = upper;

    m.def("forward_kinematics", [](const Vector5d& q){
        return so101::forwardKinematics(toAngles(q)).matrix();
    }, py::arg("q"), "4x4 world pose of the tool frame for joint angles q (5, radians).");

    m.def("joint_frames", [](const Vector5d& q){
        const so101::JointFrames frames = so101::jointFrames(toAngles(q));
        std::vector<Eigen::Matrix4d> joints;
        for(const auto& T : frames.joints) joints.push_back(T.matrix());
        return py::make_tuple(joints, frames.tool.matrix());
    }, py::arg("q"), "(list of five 4x4 joint poses, 4x4 tool pose). Each joint's z-axis is its rotation axis.");

    m.def("geometric_jacobian", [](const Vector5d& q){
        return so101::Jacobian(so101::geometricJacobian(toAngles(q)));
    }, py::arg("q"), "6x5 Jacobian: rows 0-2 tool-point linear velocity, rows 3-5 angular velocity, world frame.");

    m.def("numerical_jacobian", [](const Vector5d& q, double h){
        return so101::Jacobian(so101::numericalJacobian(toAngles(q), h));
    }, py::arg("q"), py::arg("h") = 1e-6, "Central-difference Jacobian, same layout as geometric_jacobian.");

    m.def("within_limits", [](const Vector5d& q, double tol){
        return so101::withinLimits(toAngles(q), tol);
    }, py::arg("q"), py::arg("tol") = 1e-12);

    m.def("clamp_to_limits", [](const Vector5d& q){
        return toVector(so101::clampToLimits(toAngles(q)));
    }, py::arg("q"));

    m.def("pose_error", [](const Eigen::Matrix4d& target, const Eigen::Matrix4d& current){
        const so101::PoseError e = so101::poseError(toSE3(target), toSE3(current));
        return py::make_tuple(e.position, e.orientation);
    }, py::arg("target"), py::arg("current"), "(position error, rotation vector), both in the world frame.");

    py::enum_<so101::IKMode>(m, "IKMode")
        .value("POSITION", so101::IKMode::Position)
        .value("POSE", so101::IKMode::Pose);

    py::class_<so101::IKOptions>(m, "IKOptions",
        "IK solver settings. Pass fields as keyword arguments, e.g. IKOptions(mode=IKMode.POSITION).\n"
        "Note: `locked` is copied in and out, so assign a whole list: opts.locked = [False]*4 + [True].")
        .def(py::init([](const py::kwargs& kwargs){
            py::object self = py::cast(so101::IKOptions());
            for(const auto& item : kwargs) py::setattr(self, item.first, item.second);
            return self.cast<so101::IKOptions>();
        }))
        .def_readwrite("mode", &so101::IKOptions::mode)
        .def_readwrite("position_weight", &so101::IKOptions::position_weight)
        .def_readwrite("orientation_weight", &so101::IKOptions::orientation_weight)
        .def_readwrite("position_tolerance", &so101::IKOptions::position_tolerance)
        .def_readwrite("orientation_tolerance", &so101::IKOptions::orientation_tolerance)
        .def_readwrite("initial_damping", &so101::IKOptions::initial_damping)
        .def_readwrite("min_damping", &so101::IKOptions::min_damping)
        .def_readwrite("max_damping", &so101::IKOptions::max_damping)
        .def_readwrite("max_step", &so101::IKOptions::max_step)
        .def_readwrite("stall_ratio", &so101::IKOptions::stall_ratio)
        .def_readwrite("max_iterations", &so101::IKOptions::max_iterations)
        .def_readwrite("max_restarts", &so101::IKOptions::max_restarts)
        .def_readwrite("seed", &so101::IKOptions::seed)
        .def_readwrite("locked", &so101::IKOptions::locked);

    py::class_<so101::IKResult>(m, "IKResult")
        .def_property_readonly("angles", [](const so101::IKResult& r){ return toVector(r.angles); })
        .def_readonly("success", &so101::IKResult::success)
        .def_readonly("iterations", &so101::IKResult::iterations)
        .def_readonly("restarts", &so101::IKResult::restarts)
        .def_readonly("position_error", &so101::IKResult::position_error)
        .def_readonly("orientation_error", &so101::IKResult::orientation_error)
        .def("__bool__", [](const so101::IKResult& r){ return r.success; })
        .def("__repr__", [](const so101::IKResult& r){
            return "IKResult(success=" + std::string(r.success ? "True" : "False") +
                   ", position_error=" + std::to_string(r.position_error) +
                   ", orientation_error=" + std::to_string(r.orientation_error) +
                   ", iterations=" + std::to_string(r.iterations) + ")";
        });

    m.def("solve_ik", [](const Eigen::Matrix4d& target, const Vector5d& seed, const so101::IKOptions& options){
        const Sophus::SE3d T = toSE3(target);
        const so101::JointAngles q = toAngles(seed);
        py::gil_scoped_release release;
        return so101::solveIK(T, q, options);
    }, py::arg("target"), py::arg("seed"), py::arg("options") = so101::IKOptions(),
       "Joint angles that put the tool at the 4x4 target pose, starting from seed (usually the current angles).");

    py::class_<so101::HardwareCalibration>(m, "HardwareCalibration",
        "Maps LeRobot degrees to URDF radians: q = sign * (deg - zero_deg) * pi/180.")
        .def(py::init([](const std::array<double, 5>& sign, const std::array<double, 5>& zero_deg){
            return so101::HardwareCalibration{sign, zero_deg};
        }), py::arg("sign") = std::array<double, 5>{1.0, 1.0, 1.0, 1.0, 1.0}, py::arg("zero_deg") = std::array<double, 5>{})
        .def_readwrite("sign", &so101::HardwareCalibration::sign)
        .def_readwrite("zero_deg", &so101::HardwareCalibration::zero_deg)
        .def("__repr__", [](const so101::HardwareCalibration& c){
            std::string s = "HardwareCalibration(sign=[";
            for(std::size_t i=0; i<5; ++i) s += (i ? ", " : "") + std::to_string(c.sign[i]);
            s += "], zero_deg=[";
            for(std::size_t i=0; i<5; ++i) s += (i ? ", " : "") + std::to_string(c.zero_deg[i]);
            return s + "])";
        });

    m.attr("LEROBOT_CONVENTION") = so101::kHardwareCalibration;

    m.def("from_motor_degrees", [](const Vector5d& deg, const so101::HardwareCalibration& c){
        return toVector(so101::fromMotorDegrees(toAngles(deg), c));
    }, py::arg("deg"), py::arg("calibration"),
       "LeRobot joint degrees (5, no gripper) -> URDF joint angles in radians.");

    m.def("to_motor_degrees", [](const Vector5d& q, const so101::HardwareCalibration& c){
        return toVector(so101::toMotorDegrees(toAngles(q), c));
    }, py::arg("q"), py::arg("calibration"),
       "URDF joint angles in radians -> LeRobot joint degrees to send as goal positions.");
}
