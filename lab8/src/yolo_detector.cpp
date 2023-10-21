#include "yolo_detector.h"
#include "helper_functions.hpp"
/**
 * CONSTRTUCTOR
*/
YOLODetector::YOLODetector(ros::NodeHandle &nh)
{
    yolo_detector.reset(new YOLOClient(nh, "/darknet_ros/check_for_objects", true));
    if(!yolo_detector->waitForServer(ros::Duration(5.0))){
        std::cout << "YOLO server not launched." << std::endl;
        return;
    };
    std::cout << "YOLO init!" << std::endl;
};

/**
 * detects objects in the image and returns the bboxes.
 * the function BLOCKS execution. 
*/
int YOLODetector::detect(sensor_msgs::ImageConstPtr const& image, OUT vector<darknet_ros_msgs::BoundingBox>& bboxes){
    darknet_ros_msgs::CheckForObjectsGoal goal;
    goal.image = *image;

    // code runs async
    auto state = yolo_detector->sendGoalAndWait(goal, ros::Duration(5.0));
    darknet_ros_msgs::CheckForObjectsResultConstPtr result = yolo_detector->getResult();
    
    // maybe moving should be fast?
    bboxes.insert(bboxes.begin(), result->bounding_boxes.bounding_boxes.begin(), result->bounding_boxes.bounding_boxes.end());
    return true;
}


