#pragma once
#define OUT

#include <iostream>
#include <string>
#include <vector>

#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/CheckForObjectsAction.h>

//gtsam
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point2.h>

//visualization
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include "trajectory_color.h"

//ros
#include <ros/publisher.h>
#include <ros/ros.h>

/**
 * Works for freiburg3_teddy dataset only. Converts a sensor_msgs::ImageConstPtr into a bgr OpenCV image, and returns the camera pose.
 * @param tf_listener A tf::TransformListener object that should stay alive from the beginning of your program to the end
 * @param msg The image to process
 * @return If success, OpenCV image and camera pose. Otherwise, an empty cv::Mat and pose. Check for success with `result.first.empty()`
 */
std::pair<cv::Mat, geometry_msgs::Pose> processImageAndPose(tf::TransformListener const& tf_listener, sensor_msgs::ImageConstPtr const& msg) {
    // Convert ROS msg type to OpenCV image type.
    cv::Mat img;
    try {
        img = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        return std::make_pair<cv::Mat, geometry_msgs::Pose>({}, {});
    }

    // Get the latest coordinate transform from camera to world
    tf::StampedTransform T_WC;
    std::string str = tf_listener.allFramesAsString();
    //std::cout << "transforms:\n" <<  str << std::endl;
    try {
        tf_listener.lookupTransform("world", "openni_rgb_optical_frame", ros::Time(0), T_WC);
        //tf_listener.lookupTransform("world", "openni_camera", ros::Time(0), T_WC);
    } catch (tf::TransformException &ex) {
        ROS_ERROR("%s", ex.what());
        return std::make_pair<cv::Mat, geometry_msgs::Pose>({}, {});
    }

    // convert from tf to geometry
    geometry_msgs::Pose camera_pose;
    tf::poseTFToMsg(tf::Transform(T_WC), camera_pose);
    return {std::move(img), camera_pose};
}


/**
 * Computes the centroid of a bounding box which came from a darknet_ros_msgs::CheckForObjectsResult object.
 * A CheckForObjectsResult object can be obtained from the YOLOClient (see above) by calling `client.sendGoalAndWait(goal)`
 * and then `client.getResult()`, where goal is a darknet_ros_msgs::CheckForObjectsGoal object that can be constructed directly.
 *
 * Hint: You may want to confirm that bbox.Class == "teddy bear" before calling this.
 *
 * @param bbox A bounding box of an object detection. Note that bbox.Class is a string that gives the class of object detected
 * @param img Optional. If provided, will draw the bounding box, label, and confidence level on image.
 * @return The centroid of the bounding box
 */
cv::Point2f findCentroid(darknet_ros_msgs::BoundingBox const &bbox, cv::Mat const& img = cv::Mat()) {
    cv::Point2f centroid;
    
    // Calculate centroid of YOLO detection bounding box
    centroid.x = bbox.xmin + (bbox.xmax - bbox.xmin) / 2.0f;
    centroid.y = bbox.ymin + (bbox.ymax - bbox.ymin) / 2.0f;

    // save ROI for later
    // roi = cv::Rect(cv::Point2f(bbox.xmin, bbox.ymin), cv::Point2f(bbox.xmax, bbox.ymax));

    // let's see it
    //ROS_INFO_STREAM(
    //    bbox.Class << " (" << bbox.probability << "): (" << centroid.x << "," << centroid.y 
    //               << ")");
    if (!img.empty()) {
        cv::rectangle(img, cv::Point(bbox.xmin, bbox.ymin), cv::Point(bbox.xmax, bbox.ymax), cv::Scalar(0, 0, 255), 2, 8, 0);
        cv::circle(img, centroid, 4, cv::Scalar(0, 0, 255), -1);
    }

    return centroid;
}

/**
 * Converts a bgr8 cv::Mat into a sensor_msgs::Image to be published
 * @param header The header. Construct directly or get this from bbox.header, where bbox is a darknet_ros_msgs::BoundingBox object
 * @param img The cv::Mat bgr3 image
 * @return An image message that can be published using an image_transport::Publisher object
 */
sensor_msgs::ImageConstPtr matToImageMsg(std_msgs::Header const& header, cv::Mat const& img) {
    return cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
}

/*
* returns the Quaternion with x axis pointing to "to" and z axis world up(0, 0, 1)
*/
geometry_msgs::Quaternion lookAtQuaternion(geometry_msgs::Point const& from, geometry_msgs::Point const& to){
    tf::Vector3 t(to.x - from.x, to.y - from.y , to.z - from.z);
    double yaw = atan2(t.y(), t.x());
    double pitch = atan2(-t.z(), sqrt(t.y()*t.y() + t.x()*t.x()));

    tf::Quaternion q;
    q.setRPY(0.0, pitch, yaw);
    geometry_msgs::Quaternion msg;
    msg.x = q.x();
    msg.y = q.y();
    msg.z = q.z();
    msg.w = q.w();
    return msg;
}

void DrawPoints3(const ros::Publisher& marker_pub,
                std::vector<geometry_msgs::Point>const& landmarks,
                size_t new_from,
                const TrajectoryColor& color = TrajectoryColor()) {
  static constexpr char kFrameId[] = "world";
  constexpr int32_t OLD_LANDMARKS = 0, NEW_LANDMARKS = 1;

  visualization_msgs::Marker landmark_markers;
  landmark_markers.header.frame_id = kFrameId;
  landmark_markers.header.stamp = ros::Time::now();
  landmark_markers.ns = "landmark";
  landmark_markers.pose.orientation.w = 1.0;
  landmark_markers.type = visualization_msgs::Marker::POINTS;
  landmark_markers.scale.x = 0.1;
  landmark_markers.scale.y = 0.1;
  landmark_markers.scale.z = 0.1;


  // update old markers in green(by default)
  landmark_markers.id = OLD_LANDMARKS;

  landmark_markers.color.r = color.point_color_.r_;
  landmark_markers.color.g = color.point_color_.g_;
  landmark_markers.color.b = color.point_color_.b_;
  //landmark_markers.color.a = color.point_color_.a_;
  landmark_markers.color.a = 0.5;

  //old markers to update
  landmark_markers.action = visualization_msgs::Marker::ADD;
  landmark_markers.points.assign(landmarks.begin(), landmarks.begin()+new_from);
  
  //size_t i = 0;
  //for (; i < new_from; ++i) {
  // landmark_markers.points.push_back(landmarks[i]);
  //
  if (landmark_markers.points.size())
    marker_pub.publish(landmark_markers);

  //new markers are in red
  //landmark_markers.id = NEW_LANDMARKS;
//
  //landmark_markers.color.r = 1.0;
  //landmark_markers.color.g = 0.0;
  //landmark_markers.color.b = 0.0;
  //landmark_markers.points.assign(landmarks.begin()+new_from, landmarks.end());
//
  //if(landmark_markers.points.size())
  //  marker_pub.publish(landmark_markers);

}

/*
* Draws a single elliptic marker
*/
void DrawSphere(const ros::Publisher& marker_pub,
                geometry_msgs::Point pt,
                const PointColor& color = PointColor()){
  static constexpr char kFrameId[] = "world";

  visualization_msgs::Marker marker;
  marker.header.frame_id = kFrameId;
  marker.header.stamp = ros::Time::now();
  marker.ns = "landmark";
  marker.type = visualization_msgs::Marker::SPHERE;

  marker.color.r = color.r_;
  marker.color.g = color.g_;
  marker.color.b = color.b_;
  marker.color.a = color.a_;

  marker.scale.x = 0.3;
  marker.scale.y = 0.3;
  marker.scale.z = 0.5;
  //old markers to update
  marker.action = visualization_msgs::Marker::ADD;

  marker.pose.position = pt;
  marker.pose.orientation.w = 1.0;
  marker_pub.publish(marker);

}

void DrawPoses(const ros::Publisher& pose_array_pub,
                std::vector<geometry_msgs::Pose>const& poses,
                size_t new_from) {
  static constexpr char kFrameId[] = "world";
  geometry_msgs::PoseArray pose_array;
  pose_array.header.stamp = ros::Time::now();
  pose_array.header.frame_id = kFrameId;

  pose_array.poses = std::vector<geometry_msgs::Pose>(poses);

  // Publish poses.
  pose_array_pub.publish(pose_array);
}

void DrawTrajectory(const ros::Publisher& traj_pub,
        std::vector<geometry_msgs::Pose>const& poses,
        const LineColor& color = LineColor()){
    
    static constexpr char kFrameId[] = "world";
    visualization_msgs::Marker markers;
    markers.header.frame_id = kFrameId;
    markers.header.stamp = ros::Time::now();
    markers.ns = "trajectory";

    markers.type = visualization_msgs::Marker::LINE_STRIP;
    markers.action = visualization_msgs::Marker::ADD;
    markers.color.r = color.r_;
    markers.color.g = color.g_;
    markers.color.b = color.b_;
    markers.color.a = color.a_;

    markers.points.resize(poses.size());
    for(int i = 0; i < poses.size(); ++i){
        markers.points[i] = poses[i].position;
    }

    markers.pose.orientation.w = 1.0;
    
    
    //line width
    markers.scale.x = 0.1;
    //not used?
    //markers.scale.y = 1.0;
    //markers.scale.z = 1.0;
    if(markers.points.size() > 0)
        traj_pub.publish(markers);
    
}