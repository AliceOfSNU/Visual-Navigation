#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {

class MoCapPosition3Factor : public NoiseModelFactor1<Pose3> {
 private:
  // Measurement information.
  Point3 m_;

 public:
  /**
   * Constructor
   * @param poseKey    The key of the pose.
   * @param m          Point3 measurement.
   * @param model      Noise model for Motion Capture sensor.
   */
  MoCapPosition3Factor(gtsam::Key poseKey, const gtsam::Point3& m,
                       const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor1<Pose3>(model, poseKey), m_(m) {}

  // Error function.
  // @param p 3D pose.
  // @param H optional Jacobian matrix.
  gtsam::Vector evaluateError(
      const gtsam::Pose3& p,
      boost::optional<gtsam::Matrix&> H = boost::none) const {

      if (H) {
        Matrix36 H_mat = Matrix36::Zero();
        H_mat.block<3, 3>(0, 3) = p.rotation().matrix();
        //H_mat.block<3, 3>(0, 3) = Matrix33::Identity();
        *H = H_mat;
      }

      return (Vector(3) << p.x() - m_.x(), p.y() - m_.y(), p.z() - m_.z()).finished();
  }

  ~MoCapPosition3Factor() {}
};

}  // namespace gtsam
