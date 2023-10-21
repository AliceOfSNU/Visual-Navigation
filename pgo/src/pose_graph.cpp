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
        noiseModel::Isotropic::Sigma(3, 1.0);

    const gtsam::noiseModel::Diagonal::shared_ptr pose_noise = 
    gtsam::noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(isotropic_rot_sigma), Vector3::Constant(isotropic_pos_sigma)).finished());
};

/**
 * adds a PriorFactor to the provided cam pose,
*/
//void PoseGraphOptimizer::add_pose_and_projection_factor(geometry_msgs::PoseConstPtr& cam_pose, cv::Point2f& centroid_pixel){
//    graph.add(PriorFactor<Pose3>(Symbol('x', num_cam_poses), msgToPose3(cam_pose), pose_noise));
//    graph.add(ProjectionFactor(cvToPoint2(centroid_pixel), pixel_noise, Symbol('x', num_cam_poses), Symbol('l', num_obj_pos), K));
//    num_factors++;
//}

/**
 * sets the K matrix
*/
void PoseGraphOptimizer::set_K(double fx, double fy, double s, double u0, double v0){
    K.reset(new Cal3_S2(fx, fy, s, u0, v0));
    cout << "K:" << *K << endl;
}

/*
* add a prior belief for current pose.
*/
void PoseGraphOptimizer::add_pose_prior(size_t cam_id, const Pose3& pose){
    graph.add(PriorFactor<Pose3>(Symbol('x', cam_id), pose, pose_noise));
    initialEstimate.insert(Symbol('x', cam_id), pose);
}

void PoseGraphOptimizer::add_point_prior(size_t landmark_id, const Point3& point){
    graph.add(PriorFactor<Point3>(Symbol('l', landmark_id), point, position_noise));
}

/*
* add a projection factor, and initialize estimates at the same time
*/
void PoseGraphOptimizer::add_projection_factor(size_t cam_id, size_t landmark_id, const Point2& landmark_pixel){
    graph.add(ProjectionFactor(landmark_pixel, pixel_noise, Symbol('x', cam_id), Symbol('l', landmark_id), K));
    //if first time receiving this landmark, 
    if(initialEstimate.find(Symbol('l', landmark_id)) == initialEstimate.end()){
        Pose3 cam = initialEstimate.at<Pose3>(Symbol('x', cam_id));
        Point3 estimate = Point3(landmark_pixel.x(), landmark_pixel.y(), 1.0);
        estimate = 1.0*cam.transform_from(K->matrix_inverse()*estimate);
        initialEstimate.insert(Symbol('l', landmark_id), estimate);
    }
}


Values PoseGraphOptimizer::optimize(){
  LevenbergMarquardtParams params;
  params.setVerbosity("ERROR");
  params.setAbsoluteErrorTol(1e-08);
  return LevenbergMarquardtOptimizer(graph, initialEstimate, params).optimize();
}