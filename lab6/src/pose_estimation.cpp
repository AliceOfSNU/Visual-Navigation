/*
 * @file pose_estimation.cpp
 * @brief Estimates the pose from frame to frame.
 */


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "feature_tracker.h"
#include "sift_feature_tracker.h"
#include "orb_feature_tracker.h"
#include "surf_feature_tracker.h"
#include "fast_feature_tracker.h"

#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>


#include "lab6_utils.h"
#include "pose_estimation.h"

// OpenGV
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/point_cloud/methods.hpp>

using Vector3f = Eigen::Vector3f;
using Matrix3f = Eigen::Matrix3f;
using Vector4f = Eigen::Vector4f;
using Matrix4f = Eigen::Matrix4f;
using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;
using namespace std;
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  16.485 - Fall 2021  - Lab 6 coding assignment
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//  In this code, we ask you to use the following methods for motion estimation:
//  - Nister's 5 point algorithm (2D-2D correspondences)
//  - Longuet-Higgins 8 point algorithm (2D-2D correspondences)
//  - OpenGV's 2 point algorithm (2D-2D correspondences, known rotation)
//  - Arun's 3 point algorithm (3D-3D correspondences)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_bool(use_ransac, true, "Use Random Sample Consensus.");
DEFINE_bool(scale_translation, true, "Whether to scale estimated translation to match ground truth scale");
DEFINE_int32(pose_estimator, 0,
             "Pose estimation algorithm, valid values are:"
             "0 for OpengGV's 5-point algorithm."
             "1 for OpengGV's 8-point algorithm."
             "2 for OpengGV's 2-point algorithm."
             "3 for Arun's 3-point method.");

using namespace std;
using namespace cv::xfeatures2d;
namespace enc = sensor_msgs::image_encodings;

// Global definitions.
// Mono OpenGV RANSAC problems and adapters.
// See OpenGV examples on how to use:
// https://laurentkneip.github.io/opengv/page_how_to_use.html
using RansacProblem = opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;
using Adapter = opengv::relative_pose::CentralRelativeAdapter;
using RansacProblemGivenRot = opengv::sac_problems::relative_pose::TranslationOnlySacProblem;
using AdapterGivenRot = opengv::relative_pose::CentralRelativeAdapter;
using Adapter3D = opengv::point_cloud::PointCloudAdapter;
using RansacProblem3D = opengv::sac_problems::point_cloud::PointCloudSacProblem;
using namespace opengv::sac_problems::relative_pose;

// Global variables. As a general coding style, global variables are followed
// by a trailing underscore (Google coding style).
std::shared_ptr<FeatureTracker> feature_tracker_;
ros::Publisher pub_pose_estimation_, pub_pose_gt_;
ros::Subscriber sub_cinfo_;
geometry_msgs::PoseStamped curr_pose_;
geometry_msgs::PoseStamped prev_pose_;

// See definition of CameraParams in lab6_utils.h
// We declare these global variables, and we assign values to them later in the main function
CameraParams camera_params_;
cv::Mat R_camera_body, t_camera_body;
cv::Mat T_camera_body;
geometry_msgs::Pose pose_camera_body;
tf::Transform transform_camera_body;
string stats_file_name_;
ofstream stats_file_;

void poseCallbackTesse(const nav_msgs::Odometry::ConstPtr& msg){
  // This function takes a message from the simulator of type nav_msgs::Odometry,
  // and use it to update the ground-truth pose of the drone's camera frame.
  // 
  // Note that the simulator provides the pose of the drone's body frame,
  // Therefore, we need to use the fixed transformation from the drone's camera
  // to the drone's camera to obtain the pose of the drone's camera frame.
  // 
  // In math, we want to get T_{camera}^W, but the msg provides T_{body}^W,
  // so we should do T_{camera}^W = T_{body}^W * T_{camera}^{body} to get the 
  // drone's camera pose.

  // This is T_{body}^W
  curr_pose_.pose = msg->pose.pose;

  // Convert pose message to tf::Transform
  tf::Transform current_pose;
  tf::poseMsgToTF(curr_pose_.pose, current_pose);

  // Perform the coordinate transform T_{camera}^W = T_{body}^W * T_{camera}^{body}
  // and convert to pose message
  tf::poseTFToMsg(current_pose * transform_camera_body, curr_pose_.pose);

  // publish the converted pose message so we can visualize in rViz
  curr_pose_.header.frame_id = "world";
  pub_pose_gt_.publish(curr_pose_);
}

/**
 * @brief      Compute 3D bearing vectors from pixel points
 *
 * @param[in]  pts1              Feature correspondences from camera 1
 * @param[in]  pts2              Feature correspondences from camera 2
 * @param      bearing_vector_1  Bearing vector to pts1 in camera 1
 * @param      bearing_vector_2  Bearing vector to pts2 in camera 2
 */
void calibrateKeypoints(const std::vector<cv::Point2f>& pts1,
                        const std::vector<cv::Point2f>& pts2,
                        opengv::bearingVectors_t& bearing_vector_1,
                        opengv::bearingVectors_t& bearing_vector_2) {
    //provided code.

    std::vector<cv::Point2f> points1_rect, points2_rect;
    cv::undistortPoints(pts1, points1_rect, camera_params_.K, camera_params_.D);
    cv::undistortPoints(pts2, points2_rect, camera_params_.K, camera_params_.D);

    for (auto const& pt: points1_rect){
      opengv::bearingVector_t bearing_vector(pt.x, pt.y, 1); // focal length is 1 after undistortion
      bearing_vector_1.push_back(bearing_vector.normalized());
    }

    for (auto const& pt: points2_rect){
      opengv::bearingVector_t bearing_vector(pt.x, pt.y, 1); // focal length is 1 after undistortion
      bearing_vector_2.push_back(bearing_vector.normalized());
    }
}

/**
 * @brief      Update pose estimate using previous absolue pose and estimated relative pose
 *
 * @param[in]  prev_pose         ground-truth absolute pose of previous frame
 * @param[in]  relative_pose     estimated relative pose between current frame and previous frame
 * @param      output            estimated absolute pose of current frame
 */
void updatePoseEstimate(geometry_msgs::Pose const& prev_pose, geometry_msgs::Pose const& relative_pose, geometry_msgs::Pose& output) {
  tf::Transform prev, relative;
  tf::poseMsgToTF(prev_pose, prev);
  tf::poseMsgToTF(relative_pose, relative);
  tf::poseTFToMsg(prev*relative, output); //result in world frame
}

/**
 * @brief      Given an estimated translation up to scale, return an absolute translation with correct scale using ground truth
 *
 * @param[in]  prev_pose         ground-truth absolute pose of previous frame
 * @param[in]  curr_pose         ground-truth absolute pose of current frame
 * @param      translation       estimated translation between current frame and previous frame
 */
void scaleTranslation(geometry_msgs::Point& translation, geometry_msgs::PoseStamped const& prev_pose, geometry_msgs::PoseStamped const& curr_pose) {
  if (!FLAGS_scale_translation) return;
  tf::Transform prev, curr;
  tf::poseMsgToTF(prev_pose.pose, prev);
  tf::poseMsgToTF(curr_pose.pose, curr);
  tf::Transform const relative_pose = prev.inverseTimes(curr);
  double const translation_scale = relative_pose.getOrigin().length();
  if (isnan(translation_scale) || isinf(translation_scale)) {
    ROS_WARN("Failed to scale translation");
    return;
  }
  double const old_scale = sqrt(pow(translation.x, 2) + pow(translation.y, 2) + pow(translation.z, 2));
  translation.x *= translation_scale / old_scale;
  translation.y *= translation_scale / old_scale;
  translation.z *= translation_scale / old_scale;
}


/** @brief    
    @param[in] gt_t_prev_frame ground-truth transform for previous frame.
    @param[in] gt_t_curr_frame ground-truth transform for current frame.
    @param[in] est_t_prev_frame estimated transform for previous frame.
    @param[in] est_t_curr_frame estimated transform for current frame.
*/
void evaluateRPE(const tf::Transform& gt_t_prev_frame,
                 const tf::Transform& gt_t_curr_frame,
                 const tf::Transform& est_t_prev_frame,
                 const tf::Transform& est_t_curr_frame) {

  tf::Transform const est_relative_pose = est_t_prev_frame.inverseTimes(est_t_curr_frame);
  tf::Transform const gt_relative_pose = gt_t_prev_frame.inverseTimes(gt_t_curr_frame);
  
  tf::Matrix3x3 const tf_est_relative_rot = est_relative_pose.getBasis();
  tf::Vector3 const tf_est_relative_t = est_relative_pose.getOrigin();
  tf::Matrix3x3 const tf_gt_relative_rot = gt_relative_pose.getBasis();
  tf::Vector3 const tf_gt_relative_t = gt_relative_pose.getOrigin();
  Matrix3d est_relative_rot, gt_relative_rot;
  Vector3d est_relative_t, gt_relative_t;
  tf::matrixTFToEigen(tf_est_relative_rot, est_relative_rot);
  tf::vectorTFToEigen(tf_est_relative_t, est_relative_t);
  tf::matrixTFToEigen(tf_gt_relative_rot, gt_relative_rot);
  tf::vectorTFToEigen(tf_gt_relative_t, gt_relative_t);
  gt_relative_t.normalize();
  est_relative_t.normalize();

  Vector3d err_t = (est_relative_t - gt_relative_t);
  double err_sz_t = err_t.norm();

  err_t.normalize();
  Matrix3d err_rot = (est_relative_rot - gt_relative_rot);
  double err_sz_rot = err_rot.norm();
  //cout << err_t << ", " << err_sz_rot << "\n";
  stats_file_ << err_t(0) << "," << err_t(1) << "," << err_t(2) << "," << err_sz_t << "," << err_sz_rot << endl;
}

/** @brief This function is called when a new image is published. This is
 *   where all the magic happens for this lab
 *  @param[in]  rgb_msg    RGB Camera message
 *  @param[in]  depth_msg  Depth Camera message
 */
void cameraCallback(const sensor_msgs::ImageConstPtr &rgb_msg, const sensor_msgs::ImageConstPtr &depth_msg) {

  cv::Mat bgr, depth;

  try {
    // Convert ROS msg type to OpenCV image type.
    bgr = cv_bridge::toCvShare(rgb_msg, "bgr8")->image;
    depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding)->image;
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("Could not convert rgb or depth images.");
  }

  // for drawing
  //cv::Mat view = bgr.clone();

  // cv::imshow("view", view);

  static cv::Mat prev_bgr = bgr.clone();
  static cv::Mat prev_depth = depth.clone();

  // Track features returns the 2D-2D matches between images
  // in pixels (pixel coords in image 1 -> pixel coords in image 2).
  std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> matched_kp_1_kp_2;
  try{
    feature_tracker_->trackFeatures(prev_bgr, bgr, &matched_kp_1_kp_2);
  }catch(...){
    ROS_WARN("tracking lost");
  }

  int N = matched_kp_1_kp_2.first.size();

  std::vector<cv::Point2f> pts1, pts2;
  
  cv::KeyPoint::convert(matched_kp_1_kp_2.first, pts1);
  cv::KeyPoint::convert(matched_kp_1_kp_2.second, pts2);

  opengv::bearingVectors_t bearing_vector_1, bearing_vector_2;
  calibrateKeypoints(pts1, pts2, OUT bearing_vector_1, OUT bearing_vector_2);

  // We create the central relative adapter, have a look at OpenGV's
  // documentation to understand how this class works.
  Adapter adapter_mono (bearing_vector_1, bearing_vector_2);

  // (TODO) Compute the pose estimate using the different techniques
  // here. You should fill the variable `pose_estimate` with your pose
  // estimation.
  geometry_msgs::PoseStamped pose_estimation;
  tf::poseTFToMsg(tf::Pose::getIdentity(), pose_estimation.pose);

  // initialize a relative pose estimate, you will update relative_pose_estimate using the following algorithms
  geometry_msgs::Pose relative_pose_estimate = pose_estimation.pose;

  switch(FLAGS_pose_estimator) {
  case 0: {
    // ******************************** 5-point Nister minimal solver *********************************

    // First, ensure we have the minimal number of correspondences to compute the relative pose.
    static constexpr size_t min_nr_of_correspondences = 5;
    if (adapter_mono.getNumberCorrespondences() >= min_nr_of_correspondences) {
      if (!FLAGS_use_ransac) {
        opengv::essentials_t fivept_nister_essentials = opengv::relative_pose::fivept_nister( adapter_mono );
        opengv::transformation_t best_transformation;

        cv::Mat R, t;
        //Matrix3d Re; Vector3d te;
        int num_inliers = extractPose(fivept_nister_essentials, pts1, pts2, camera_params_, OUT R, OUT t);

        // to geometry_msgs
        relative_pose_estimate = cv2Pose(R, t);
      } else {
        shared_ptr<CentralRelativePoseSacProblem> solver_ptr(
            new CentralRelativePoseSacProblem(adapter_mono, CentralRelativePoseSacProblem::NISTER) 
        );
#if 0
        opengv::sac::Ransac<CentralRelativePoseSacProblem> ransac;
        ransac.sac_model_ = solver_ptr;
        const float threshold = 0.005;
        const int max_iters = 50; 

        ransac.threshold_ = threshold;
        ransac.max_iterations_ = max_iters;
        
        // should be in a try-catch

        ransac.computeModel(2);
        opengv::transformation_t best_transformation;
        best_transformation.block<3, 3>(0, 0) = Matrix3d::Identity();
        best_transformation.col(3) = Vector3d::Zero();

        best_transformation = ransac.model_coefficients_;

#endif

        opengv::transformation_t best_transformation;
        opengv::essentials_t fivept_nister_essentials;

        // RANSAC
        int num_iters = 0;
        const float threshold = 0.01;
        const int max_iters = 50; 

        int num_inliers = 0;
        const int ns = bearing_vector_1.size();
        const int T = ns * 0.60; //96% error_free
        double best_error = 2.0;
        vector<cv::Point2f> corr1(min_nr_of_correspondences), corr2(min_nr_of_correspondences);
        while(num_iters < max_iters){

          // 1. select a subset
          set<int> indices; vector<int> inlier_indices{};
          while(indices.size() < min_nr_of_correspondences){
            indices.insert(rand()%ns);
          }
          vector<int> vindices;
          for(int x: indices) {
            vindices.push_back(x);
            corr1.emplace_back(pts1[x]);
            corr2.emplace_back(pts2[x]);
          }
          // 2. calculate
          
          fivept_nister_essentials = opengv::relative_pose::fivept_nister( adapter_mono, vindices);

          // 3. calculate inliers build consensus set
          cv::Mat Rcv, tcv;
          int retval = extractPose(fivept_nister_essentials, corr1, corr2, camera_params_, OUT Rcv, OUT tcv);
          if(retval <= 0){
            num_iters ++; continue;
          }

          Matrix3d R = Matrix3d::Identity() ; Vector3d t = Vector3d::Zero();
          for(int i = 0; i < 3; ++i){
            for(int j = 0; j < 3; ++j){
              R(i, j) = Rcv.at<double>(i, j);
            }
          }

          for(int i = 0; i < 3; ++i){
            t(i) = tcv.at<double>(i);
          }


          for(int k = 0; k < bearing_vector_1.size(); ++k){                
              double error = 0.0;

              // error calculating routine
              Vector3d y1 = bearing_vector_1[k];
              Vector3d y2 = bearing_vector_2[k];
              Vector3d Ry2 = R*y2;

              // TODO: invert signs to match opengv conventions
              Eigen::Matrix2d C;
              C << y1.dot(y1), -y1.dot(Ry2),
                  y1.dot(Ry2), -Ry2.dot(Ry2); //invertible. skew - symmetric with nonzero diagonal.
              Eigen::Vector2d td; 
              td << t.dot(y1), t.dot(Ry2);
              Eigen::Vector2d d = C.inverse()*td;
              Vector3d y2_proj = (d(0) * y1 + d(1) * Ry2 + t) / 2;

              error += 1 - y2.dot(y2_proj.normalized());
              Vector3d y1_proj = R.transpose()*(y2_proj - t);
              error += 1 - y1.dot(y1_proj.normalized());
              error /= 2;
              //end error calculation

              if(error < threshold) {
                inlier_indices.push_back(k);
                best_error = error;
              }
          } //END index loop
        
          num_inliers = inlier_indices.size();
          // 4. if inliers > T, accept and break
          if(num_inliers > T){
            fivept_nister_essentials = opengv::relative_pose::fivept_nister( adapter_mono, inlier_indices );
            break;
          }
          num_iters++;
        }

        if(num_iters >= max_iters){
          std::cout << "RANSAC reached maximum iterations" << endl;
        }else{
          std::cout << "ransac terminated after "<< num_iters << " iterations with " << num_inliers << " inliers, error:" << best_error  << endl;
        }

        cv::Mat Rcv, tcv;
        extractPose(fivept_nister_essentials, pts1, pts2, camera_params_, OUT Rcv, OUT tcv);
        relative_pose_estimate = cv2Pose(Rcv, tcv);
        
      }
    } else {
      ROS_WARN("Not enough correspondences to compute pose estimation using"
               " Nister's algorithm.");
    }
    break;
  }
  case 1: {
    // ******************************** 8-point Method *********************************
    static constexpr size_t min_nr_of_correspondences = 8;

    if (adapter_mono.getNumberCorrespondences() >= min_nr_of_correspondences) {
      if (!FLAGS_use_ransac) {
        // Without RANSAC [OPTIONAL]
        // this actually does not require RANSAC


        // Consider using the "extractPose" function provided to obtain inliers, rotation, and translation
        // Consider using "cv2Pose(Rmat, tmat)" to recover the rotation Rmat and translation tmat
        Eigen::MatrixXd A = Eigen::MatrixXd(bearing_vector_1.size(), 9); 
        Eigen::VectorXd vecE = Eigen::VectorXd::Zero(9);
        for(int i = 0; i < bearing_vector_1.size(); ++i){
          A.block<1, 3>(i, 0) = bearing_vector_2[i](0) * bearing_vector_1[i].transpose();
          A.block<1, 3>(i, 3) = bearing_vector_2[i](1) * bearing_vector_1[i].transpose();
          A.block<1, 3>(i, 6) = bearing_vector_2[i](2) * bearing_vector_1[i].transpose();
        }
        //the matrix is not huge.. so faster to use JacobiSVD instead of bcdSVD.
        Eigen::JacobiSVD<Eigen::MatrixXd> SVD(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        //Eigen::VectorXd sol = SVD.solve(Eigen::VectorXd::Zero(9));
        Eigen::Matrix<double,9,1> sol = SVD.matrixV().col(8);
        
        Matrix3d EE;
        EE.col(0) = sol.segment(0, 3);
        EE.col(1) = sol.segment(3, 3);
        EE.col(2) = sol.segment(6, 3);
        EE.transposeInPlace();

        // Essential space projection
        Eigen::JacobiSVD<Eigen::MatrixXd> SVD2(EE, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        Matrix3d S = Eigen::Matrix3d::Zero();
        S(1, 1) = S(0, 0) = (SVD2.singularValues()[0] + SVD2.singularValues()[1])/2;
        Matrix3d E = SVD2.matrixU()*S*SVD2.matrixV().transpose();

#if 0
        // calculate R and t from Essential
        Matrix3d R; Vector3d t;

        Matrix3d R_z;
        R_z << 0.0, -1.0, 0.0,
              1.0, 0.0, 0.0,
              0.0, 0.0, 1.0; 

        // the rotation matrices are the same even with -E instead of E.
        Matrix3d Rp = SVD2.matrixU() * R_z * SVD2.matrixV().transpose();
        Matrix3d Rn = SVD2.matrixU() * R_z.transpose() * SVD2.matrixV().transpose();

        // however we do require them to be in SO(3) so divide by Z(2)
        if(Rp.determinant() < 0) Rp = -Rp;
        if(Rn.determinant() < 0) Rn = -Rn;

        t = SVD2.singularValues()[0] * SVD2.matrixU().col(2);

        vector<pair<Matrix3d, Vector3d>> configs(4);
        configs[0] = {Rp, t};
        configs[1] = {Rp, -t};
        configs[2] = {Rn, t};
        configs[3] = {Rn, -t}; 

        // we try to measure the quality(see openGV impl)
        double best_error = 1E10; int best_config = 0;
        for(int i = 0; i < 4; ++i){
          Matrix3d R; Vector3d t;
          tie(R, t) = configs[i];
          double error = 0.0;
          for(int k = 0; k < bearing_vector_1.size(); ++k){
            Vector3d y1 = bearing_vector_1[k];
            Vector3d y2 = bearing_vector_2[k];
            Vector3d Ry1 = R*y1;

            // the essential constraints may not be satisfied EXACTLY because we performed least squares.
            // so we have to project.
            Eigen::Matrix2d C;
            C << y2.dot(y2), -y2.dot(Ry1),
                y2.dot(Ry1), -Ry1.dot(Ry1); //invertible. skew - symmetric with nonzero diagonal.
            Eigen::Vector2d td; 
            td << t.dot(y2), t.dot(Ry1);
            Eigen::Vector2d d = C.inverse()*td;
            Vector3d y2_proj = (d(0) * y2 + d(1) * Ry1 + t) / 2;

            error += 1 - y2.dot(y2_proj.normalized());
            Vector3d y1_proj = R.transpose()*(y2_proj - t);
            error += 1 - y1.dot(y1_proj.normalized());
          }
          if(error < best_error) best_error = error, best_config = i;
        }
        tie(R, t) = configs[best_config];
        opengv::transformation_t best_transformation;
        
        best_transformation.col(3) = -t;
        best_transformation.block<3, 3>(0, 0) = R.transpose();

        relative_pose_estimate = eigen2Pose(best_transformation);

#endif
        // recoverPose has similar performance to my code..
        // so my impl is not the problem... whyyyy???

        cv::Mat Emat, R, t;
        cv::eigen2cv(E, Emat);
        int inliers = cv::recoverPose(Emat, pts1, pts2, camera_params_.K, R, t);
        relative_pose_estimate = cv2Pose(R, t);

      } else {
        shared_ptr<CentralRelativePoseSacProblem> solver_ptr(new CentralRelativePoseSacProblem(adapter_mono, CentralRelativePoseSacProblem::EIGHTPT));

        opengv::sac::Ransac<CentralRelativePoseSacProblem> ransac;
        ransac.sac_model_ = solver_ptr;
        const float threshold = 0.01; 
        const int max_iters = 400; //tune these parameters
        ransac.threshold_ = threshold;
        ransac.max_iterations_ = max_iters;
        
        // should be in a try-catch
        ransac.computeModel();
        opengv::transformation_t best_transformation = ransac.model_coefficients_;

        // to geometry_msgs
        relative_pose_estimate = eigen2Pose(best_transformation);

      }
    } else {
      ROS_WARN("Not enough correspondences to compute pose estimation using"
               " Longuet-Higgins' algorithm.");
    }
    break;
  }
  case 2: {
    // ******************************** 2-point Method(Rotation-only) *********************************

    static constexpr size_t min_nr_of_correspondences = 2;
    if (adapter_mono.getNumberCorrespondences() >= min_nr_of_correspondences) {
      
      // Obtain the rotation part from ground-truth
      tf::Transform curr_frame, prev_frame;
      tf::poseMsgToTF(curr_pose_.pose, curr_frame);
      tf::poseMsgToTF(prev_pose_.pose, prev_frame);
      Eigen::Matrix3d rotation;
      tf::matrixTFToEigen(prev_frame.inverseTimes(curr_frame).getBasis(), rotation);
      adapter_mono.setR12(rotation);

      // We only estimate the translation part
      if (!FLAGS_use_ransac) {
        // Without RANSAC (OPTIONAL)
        // ***************************** begin solution *****************************
        


        // ***************************** end solution *****************************
      } else {
        // (TODO) With RANSAC
        // ***************************** begin solution *****************************
        


        // ***************************** end solution *****************************
      }
    } else {
      ROS_WARN("Not enough correspondences to estimate relative translation using 2pt algorithm.");
    }
    break;
  }
  case 3: {
    // ******************************** 3-point Method *********************************

    // Scale the bearing vectors to point clouds, by querying the depth values of each keypoint
    for (int i=0; i<N; i++) {

      // Use the pixel locations of the keypoints to query depth in the depth image  
      double d1 = double( prev_depth.at<float>(std::floor(pts1[i].y), std::floor(pts1[i].x)) ) ;
      double d2 = double( depth.at<float>(std::floor(pts2[i].y), std::floor(pts2[i].x)) ) ;

      // Normalize the bearing vectors such that the last entry is 1
      bearing_vector_1[i] /= bearing_vector_1[i](2,0);
      bearing_vector_2[i] /= bearing_vector_2[i](2,0);

      // Scale the bearing vectors so that the last entry is equal to depth
      bearing_vector_1[i] *= d1;
      bearing_vector_2[i] *= d2;
    }

    // OpenGV PointCloud 3D-3D Adapter
    opengv::points_t cloud_1, cloud_2;
    for (auto i = 0ul; i < bearing_vector_1.size(); i++) {
        cloud_1.push_back(bearing_vector_1[i]);
        cloud_2.push_back(bearing_vector_2[i]);
    }

    Adapter3D adapter_3d(cloud_1, cloud_2);

    static constexpr int min_nr_of_correspondences = 3; 
    if (adapter_3d.getNumberCorrespondences() >= min_nr_of_correspondences) {
      if (!FLAGS_use_ransac){
        // Custom implementation of ARUN 3pts, but isn;t it just ICP
        // Code by AliceOfSNU, SEP 2023

        //bearing vectors are scaled at this point. not unit f.
        Vector3d centroid1 = Vector3d::Zero(), centroid2 = Vector3d::Zero();
        assert(bearing_vector_1.size() == bearing_vector_2.size());
        for(int i = 0; i < bearing_vector_1.size(); ++i){
          centroid1 += bearing_vector_1[i];
          centroid2 += bearing_vector_2[i];
        }
        centroid1 /= bearing_vector_1.size();
        centroid2 /= bearing_vector_2.size();

        // covariance matrix S. if non-identical weights, do XWY*
        Matrix3d COV = Matrix3d::Zero();
        for(int i = 0; i < bearing_vector_1.size(); ++i){
          COV += (bearing_vector_1[i] - centroid1)*(bearing_vector_2[i] - centroid2).transpose();
        }

        // compute matrix R s.t. bv2 = R*bv1 in world frame.
        Matrix3d R = Matrix3d::Identity(); 
        
        // solve for R = V(SIGMA)U* where SIGMA is I or I with last entry == -1
        Eigen::JacobiSVD<Eigen::MatrixXd> svd (COV, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixV() * svd.matrixU().transpose();
        if(R.determinant() < 0) {
          Matrix3d V2 = svd.matrixV();
          V2.col(2) *= -1; //originally something like diag(1, 1, 1, 1,.... -1) in the middle
          R = V2 * svd.matrixU().transpose();
        }

        Vector3d t = centroid1 - R*centroid2;

        opengv::transformation_t best_transformation = Eigen::MatrixXd::Zero(3, 4);
        best_transformation.block<3, 3>(0,0) = R;
        best_transformation.col(3) = t;

        // to geometry_msgs
        relative_pose_estimate = eigen2Pose(best_transformation);

      } else{
        //just use opengv
        // create a 3D-3D adapter

        opengv::sac::Ransac<opengv::sac_problems::point_cloud::PointCloudSacProblem> ransac;
        shared_ptr<opengv::sac_problems::point_cloud::PointCloudSacProblem>
            relposeproblem_ptr(
            new opengv::sac_problems::point_cloud::PointCloudSacProblem(adapter_3d) );
        // run ransac
        ransac.sac_model_ = relposeproblem_ptr;
        ransac.threshold_ = 0.01; // adjust this <TODO>
        ransac.max_iterations_ = 50; // adjust this <TODO>
        ransac.computeModel(0);
        // return the result
        opengv::transformation_t best_transformation = ransac.model_coefficients_;
        relative_pose_estimate = eigen2Pose(best_transformation);

      }
    } else {
      ROS_WARN("Not enough correspondences to estimate absolute transform using Arun's 3pt algorithm.");
    }
    break;
  }
  default: {
    ROS_ERROR("Wrong pose_estimator flag!");
  }
  }

  // scale the relative pose estimate (for 5-pt, 2-pt and 8-pt, the translation is up to scale)
  if (FLAGS_pose_estimator < 3) {
    scaleTranslation(relative_pose_estimate.position, prev_pose_, curr_pose_);
  }

  // pose estimate w.r.t. ground truth
  // pose_estimation.pose is in world frame
  updatePoseEstimate(prev_pose_.pose, relative_pose_estimate, pose_estimation.pose);

  // Compute tf::Transform in order to do actual pose operations.
  tf::Transform gt_t_prev_frame, gt_t_curr_frame;
  tf::Transform est_t_prev_frame, est_t_curr_frame;
  tf::poseMsgToTF(pose_estimation.pose, est_t_curr_frame);
  tf::poseMsgToTF(curr_pose_.pose, gt_t_curr_frame);
  tf::poseMsgToTF(prev_pose_.pose, est_t_prev_frame);
  tf::poseMsgToTF(prev_pose_.pose, gt_t_prev_frame);

  // Evaluate pose errors
  evaluateRPE(gt_t_prev_frame, gt_t_curr_frame,est_t_prev_frame,est_t_curr_frame);

  pose_estimation.header.frame_id = "world";
  pub_pose_estimation_.publish(pose_estimation);

  // Save for next iteration
  prev_bgr = bgr.clone();
  prev_depth = depth.clone();
  prev_pose_ = curr_pose_;
}

/**
 * @function main
 * @brief Main function
 */
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();

  ros::init(argc, argv, "keypoint_trackers");
  ros::NodeHandle local_nh("~");

  // start logging
  if (local_nh.getParam("path_to_rpelog", stats_file_name_)) {
    ROS_INFO("Got param: %s", stats_file_name_.c_str());
  } else {
    ROS_FATAL("path to rpelog must be provided");
    return false;
  }
  std::cout << "logging at " << stats_file_name_ << endl;
  stats_file_.open(stats_file_name_);

  // populate camera intrinsics and distortion
  camera_params_.K = cv::Mat::zeros(3, 3, CV_64F);
  camera_params_.K.at<double>(0,0) = 415.69219381653056;
  camera_params_.K.at<double>(1,1) = 415.69219381653056;
  camera_params_.K.at<double>(0,2) = 360.0;
  camera_params_.K.at<double>(1,2) = 240.0;
  camera_params_.D = cv::Mat::zeros(cv::Size(5,1),CV_64F);

  // Constant extrinsics of the camera wrt the body, T_{camera}^{body}
  T_camera_body = cv::Mat::zeros(cv::Size(4,4),CV_64F);
  T_camera_body.at<double>(0,2) = 1.0;
  T_camera_body.at<double>(1,0) = -1.0;
  T_camera_body.at<double>(1,3) = 0.05;
  T_camera_body.at<double>(2,1) = -1.0;
  T_camera_body.at<double>(3,3) = 1.0;
  // convert to geometry_msgs::Pose
  R_camera_body = T_camera_body(cv::Range(0,3),cv::Range(0,3));
  t_camera_body = T_camera_body(cv::Range(0,3),cv::Range(3,4));
  pose_camera_body = cv2Pose(R_camera_body,t_camera_body);
  // convert to tf transform
  tf::poseMsgToTF(pose_camera_body, transform_camera_body);

  feature_tracker_.reset(new OrbFeatureTracker());

  // Subscribe to drone pose estimation.
  auto pose_sub = local_nh.subscribe("/ground_truth_pose", 10, poseCallbackTesse);

  // Subscribe to rgb and depth images.
  image_transport::ImageTransport it(local_nh);
  image_transport::SubscriberFilter sf_rgb(it, "/rgb_images_topic", 1);
  image_transport::SubscriberFilter sf_depth(it, "/depth_images_topic", 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sf_rgb, sf_depth);
  sync.registerCallback(cameraCallback);

  // Advertise drone pose.
  pub_pose_gt_ = local_nh.advertise<geometry_msgs::PoseStamped>("/gt_camera_pose", 1);
  pub_pose_estimation_ = local_nh.advertise<geometry_msgs::PoseStamped>("/camera_pose", 1);

  while (ros::ok()) {
    ros::spinOnce();
    cv::waitKey(1);
  }
  cv::destroyAllWindows();  
  stats_file_.close();
  std::cout << "writing ends" << endl;
  return EXIT_SUCCESS;
}
