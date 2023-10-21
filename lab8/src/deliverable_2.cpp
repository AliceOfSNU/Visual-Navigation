// ROS headers.
#include <iostream>
#include <vector>
#include <deque>


#include <ros/ros.h>
#include <ros/publisher.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>

#include <std_msgs/ColorRGBA.h>
#include "yolo_detector.h"
#include <pgo/OptimizationRequest.h>
#include <pgo/OptimizationResult.h>
#include "helper_functions.hpp"
#include "trajectory_color.h"
#include <glog/logging.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  DELIVERABLE 2 | Object Localization
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

//global storage(move to heap if large)
struct observation{
  size_t frame_id;
  geometry_msgs::Pose cam_pose;
  // (cam_id, landmark_id, centroid_estimate)
  vector<size_t> cam_ids;
  vector<size_t> landmarks_ids;
  vector<cv::Point2f> landmarks_pixels;
};

pgo::OptimizationRequest req;
bool pgo_available = true;
bool has_data = false;

size_t poses_new_index = 0;
size_t landmarks_new_index = 0;
//holds gt poses from rosbag
vector<geometry_msgs::Pose> gt_poses;
//updated to optimized poses by backend callback.
vector<geometry_msgs::Pose> poses;
vector<geometry_msgs::Point> landmarks;

// global publishers
ros::Publisher optimal_traj_lines_pub, gt_pose_array_pub, optimal_landmarks_pub, optimal_pose_array_pub;
ros::Publisher pgo_pub;

// functions
void send_optimiztion_request(){
    cout << "[dev2] sending requests with " << req.landmarks_ids.size() << endl;
    
    pgo_pub.publish(req);
    pgo_available = false;
    req = pgo::OptimizationRequest();
};

void optimizer_callback(pgo::OptimizationResultConstPtr const& msg){
    // update datastore
    cout << "[dev2] received results with " << msg->poses.size() << "poses and " << msg->landmarks.size() << "landmarks" << endl;
    pgo_available = true;

    poses_new_index = poses.size();
    poses.resize(msg->poses.size());
    for(geometry_msgs::PoseStamped const& posestamped : msg->poses){
      //poses[posestamped.header.seq] = posestamped.pose;
      poses[posestamped.header.seq].position = posestamped.pose.position;
      poses[posestamped.header.seq].orientation = lookAtQuaternion(posestamped.pose.position, msg->landmarks[0].point);
    }

    landmarks_new_index = landmarks.size();
    landmarks.resize(msg->landmarks.size());
    for(geometry_msgs::PointStamped const& pointstamped: msg->landmarks){
      landmarks[pointstamped.header.seq] = pointstamped.point;
    }

    // visualize
    DrawSphere(optimal_landmarks_pub, landmarks[0], PointColor(1.0, 0.0, 1.0, 0.5));
    DrawPoses(optimal_pose_array_pub, poses, poses_new_index);
    //DrawPoses(gt_pose_array_pub, gt_poses, 0); //draw gt for debugging
    DrawTrajectory(optimal_traj_lines_pub, poses, LineColor(0.0, 1.0, 0.0, 0.5));

    // send optimization request up to current frame.
    if(pgo_available && has_data){
      send_optimiztion_request();
    }
    
  };


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();

  // Init ROS node.
  ros::init(argc, argv, "deliverable_2");
  ros::NodeHandle local_nh("");

  // ROS publishers for poses and landmarks.
  // rviz
  optimal_traj_lines_pub = local_nh.advertise<visualization_msgs::Marker>("/trajectory_lines", 10, true);
  gt_pose_array_pub = local_nh.advertise<geometry_msgs::PoseArray>("/gt_trajectory", 10, true);
  optimal_landmarks_pub =local_nh.advertise<visualization_msgs::Marker>("/landmarks_optimal", 10, true);
  optimal_pose_array_pub = local_nh.advertise<geometry_msgs::PoseArray>("/poses_optimal", 10, true);                             
  
  tf::StampedTransform transform;
  tf::TransformListener tf_listener;
  image_transport::ImageTransport img_tr(local_nh);

  // create detector
  std::shared_ptr<YOLODetector> detector;
  detector.reset(new YOLODetector(local_nh));
  landmarks.resize(1); //we are tracking one landmark.

  ros::Subscriber pgo_sub = 
    local_nh.subscribe("/backend/optim_result", 10, optimizer_callback);

  pgo_pub = local_nh.advertise<pgo::OptimizationRequest>("/backend/optim_request", 1);
  
  size_t frameId(0); 

  auto callback = [&](const sensor_msgs::ImageConstPtr &msg) {
    std::pair<cv::Mat, geometry_msgs::Pose> img_and_pose = processImageAndPose(tf_listener, msg);
    static int initial_req = 1;

    // push cam pose
    poses.push_back(img_and_pose.second);
    gt_poses.push_back(img_and_pose.second);
    req.cam_poses.push_back(img_and_pose.second);
    req.cam_ids.push_back(frameId);

    // detect bounding boxes
    vector<darknet_ros_msgs::BoundingBox> bboxes;
    bool is_detected = detector->detect(msg, OUT bboxes);
    
    // iterate over bboxes and push teddy bears
    for (darknet_ros_msgs::BoundingBox const& bbox: bboxes){
      if(bbox.Class == "teddy bear"){
        req.association_cam_ids.push_back(frameId);
        req.landmarks_ids.push_back(0);

        // extract centroid
        cv::Point2f centroid = findCentroid(bbox);
        geometry_msgs::Point pt;
        pt.x = centroid.x, pt.y = centroid.y, pt.z = 0.0f;
        req.landmarks_pixels.push_back(pt);

        cout << "image h=" << msg->height << ", w=" << msg->width << ": x=" << pt.x << ", y = " << pt.y << endl; 
      }
    }

    
    cout <<"total" << req.landmarks_ids.size() << "landmarks" << endl;
    if(req.landmarks_ids.size() > 0) has_data = true;
    if(has_data && initial_req) {
      initial_req = false;
      send_optimiztion_request();
    }

    frameId++;
  };
  
  image_transport::Subscriber im_sub = img_tr.subscribe("/camera/rgb/image_color", 10, callback);
  //ros::Subscriber cam_sub = local_nh.subscribe("/cam/rgb/camera_info", 100, [](const sensor_msgs::CameraInfoConstPtr &msg){});
  // he K matrix can be used for the gtsam::Cal3_S2 part of a gtsam::GenericProjectionFactor

  // ROS spin until killed.
  ros::spin();

  return 0;
}
