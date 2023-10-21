#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point2.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <opencv2/core.hpp>

using namespace gtsam;
/**
 * converts a geometry_msg pose to gtsam::Pose3
*/
inline Pose3 msgToPose3(geometry_msgs::Pose const& msg){
    geometry_msgs::Quaternion const& q = msg.orientation;
    geometry_msgs::Point const& p = msg.position;

    //copy ellision or move.
    return Pose3(Rot3(Quaternion(q.w, q.x, q.y, q.z)), Point3(p.x, p.y, p.z));
}

/**
 * converts a geometry_msg point to gtsam::Point3
*/
inline Point3 msgToPoint3(geometry_msgs::Point const& msg){
    return Point3(msg.x, msg.y, msg.z);
}

/**
 * converts a geometry_msg point to gtsam::Point3
*/
inline Point2 msgToPoint2(geometry_msgs::Point const& msg){
    return Point2(msg.x, msg.y);
}

/**
* converts a cv2 Point2f to gtsam::Point2
*/
inline Point2 cvToPoint2(cv::Point2f const& cvpt2){
    return Point2(cvpt2.x, cvpt2.y);
}

inline geometry_msgs::Point pointToMsg(Point3 p){
    geometry_msgs::Point msg;
    msg.x = p.x();
    msg.y = p.y();
    msg.z = p.z();
    return msg;
}