#pragma once
#define OUT
#include <queue>
// Darknet and actions
#include <actionlib/action_definition.h>
#include <actionlib/client/simple_action_client.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/CheckForObjectsAction.h>

// CV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

// A YOLOClient (defined below) makes it easier to get detections in an image from YOLO
// You can construct one with something like: `YOLOClient yolo_detector("darknet_ros/check_for_objects", true)`
using YOLOClient = actionlib::SimpleActionClient<darknet_ros_msgs::CheckForObjectsAction>;
using namespace std;

ACTION_DEFINITION(darknet_ros_msgs::CheckForObjectsAction) // defines a bunch of actionlib types that may be useful

class YOLODetector{
    public:
    YOLODetector(ros::NodeHandle &nh);
    ~YOLODetector() = default;
    int detect(sensor_msgs::ImageConstPtr const& image, OUT vector<darknet_ros_msgs::BoundingBox>& bboxes);


    private:
    shared_ptr<YOLOClient> yolo_detector;
    queue<pair<size_t, darknet_ros_msgs::BoundingBoxes>> detection_results_q;

};