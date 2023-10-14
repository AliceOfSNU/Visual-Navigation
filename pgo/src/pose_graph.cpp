#include "pgo/pose_graph.h"

/**
 *  @constructor reates all the noise models 
*/
PoseGraphOptimizer::PoseGraphOptimizer(double pos_sigma, double rot_sigma, double pix_sigma)
:isotropic_pix_sigma(pix_sigma), isotropic_pos_sigma(pos_sigma), isotropic_rot_sigma(rot_sigma)
{
    std::cout <<"pgo optimizer init!" << endl;
    noiseModel::Isotropic::shared_ptr pixel_noise =
        noiseModel::Isotropic::Sigma(2, isotropic_pix_sigma);  // one pixel in u and v

    noiseModel::Isotropic::shared_ptr position_noise =
        noiseModel::Isotropic::Sigma(3, isotropic_pos_sigma);

    const gtsam::noiseModel::Diagonal::shared_ptr pose_noise = 
    gtsam::noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(isotropic_rot_sigma), Vector3::Constant(isotropic_pos_sigma)).finished());
};

/**
 * adds a PriorFactor to the provided cam pose,
*/
void PoseGraphOptimizer::add_pose_and_projection_factor(geometry_msgs::PoseConstPtr& cam_pose, cv::Point2f& centroid_pixel){
    graph.add(PriorFactor<Pose3>(Symbol('x', num_cam_poses), msgToPose3(cam_pose), pose_noise));
    graph.add(ProjectionFactor(cvToPoint2(centroid_pixel), pixel_noise, Symbol('x', num_cam_poses), Symbol('l', num_obj_pos), K));
    num_factors++;
}

