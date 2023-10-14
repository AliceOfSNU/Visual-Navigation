#include <vector>

#include <ros/ros.h>
#include <ros/publisher.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
//gtsam
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/ProjectionFactor.h>

#include "pgo/pose_graph.h"
#include <pgo/OptimizationRequest.h>
#include <pgo/OptimizationResult.h>

using namespace std;
using namespace gtsam;

// A projection factor well suited for tracking a point projected from a camera image
// (HINT) A gtsam::Cal3_S2 object represents camera intrinsics (see https://gtsam.org/doxygen/a02852.html#ae7de8f587615c7b0909c06df658e96e5)
using ProjectionFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;
int main(int argc, char** argv) {

  // Init ROS node.
  ros::init(argc, argv, "pgo_node");
  ros::NodeHandle local_nh("");

  shared_ptr<PoseGraphOptimizer> opt;
  opt.reset(new PoseGraphOptimizer(0.1, 0.1, 10.0));

  ros::Publisher pub = local_nh.advertise<pgo::OptimizationResult>("/backend/optim_result", 1);
  auto callback = [&](const pgo::OptimizationRequestConstPtr &msg) {
    cout << "[pgo] received cam pose" << msg->cam_pose.position << endl;
    cout << "[pgo] received landmark position" << msg->landmarks_pixels[0]  << endl;
    //test publish
    pgo::OptimizationResult res;
    res.landmarks.resize(1);
    res.poses.resize(1);
    pub.publish(res);
  };

  ros::Subscriber sub = local_nh.subscribe<pgo::OptimizationRequest>("/backend/optim_request", 10, callback);
  ros::spin();
}