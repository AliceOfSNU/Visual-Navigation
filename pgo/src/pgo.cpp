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
  opt.reset(new PoseGraphOptimizer(0.05, 0.05, 10.0));

  opt->set_K(537.96032, 539.56766, 0.0, 319.18364, 247.05382);
  ros::Publisher pub = local_nh.advertise<pgo::OptimizationResult>("/backend/optim_result", 1);
  
  auto callback = [&](const pgo::OptimizationRequestConstPtr &msg) {
    // variables to keep count of landmarks and cameras
    static int landmarks_cnt = 0, poses_cnt = 0;
    static int needs_init = true;


    cout << "[pgo] received " << msg->cam_poses.size() << " cam poses, with " << msg->landmarks_pixels.size() << " landmarks" << endl;
    
    //add cam pose priors
    for(int i = 0; i < msg->cam_poses.size();++i) {
      poses_cnt++;
      opt->add_pose_prior(msg->cam_ids[i], msgToPose3(msg->cam_poses[i]));
    }

    //add observation factors
    for(int i = 0; i < msg->landmarks_ids.size(); ++i){
      //if a new landmark, increment count
      if(!opt->has_key(gtsam::Symbol('l', msg->landmarks_ids[i]))) {
        landmarks_cnt++;
      }
      opt->add_projection_factor(msg->association_cam_ids[i], msg->landmarks_ids[i], msgToPoint2(msg->landmarks_pixels[i]));
    }
    cout << "[pgo] total " << poses_cnt << " cam poses, with " << landmarks_cnt << " landmarks" << endl;

    // needs one additional constraint on a single landmark position
    // this is not initial estimate.
    if(needs_init){
       Pose3 cam = msgToPose3(msg->cam_poses[0]);
       Cal3_S2 K = opt->get_K();
       Point2 pt2 = msgToPoint2(msg->landmarks_pixels[0]);
       Point3 estimate = Point3(pt2.x(), pt2.y(), 1.0);
       estimate = 1.0*cam.transform_from(K.matrix_inverse()*estimate);
       
       cout << estimate << endl;
       opt->add_point_prior(msg->landmarks_ids[0], estimate);
       needs_init = false;
    }

    //opt->print_graph();

    //optimize
    Values result = opt->optimize();
    //result.print("[pgo] Optimization results:\n");
    cout << "final error = " << opt->getError(result) << endl;

    //send result
    pgo::OptimizationResult res;
    for(int i = 0; i < poses_cnt; ++i){
      geometry_msgs::PoseStamped msg;
      msg.header.seq = i;
      Pose3 const& pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', i));
      Quaternion const& q = pose.rotation().toQuaternion();
      msg.pose.orientation.w = q.w();
      msg.pose.orientation.x = q.x();
      msg.pose.orientation.y = q.y();
      msg.pose.orientation.z = q.z();

      Point3 const& p = pose.translation();
      msg.pose.position.x = p.x();
      msg.pose.position.y = p.y();
      msg.pose.position.z = p.z();

      res.poses.push_back(msg);
    }
    cout << "[pgo] sending result with "<< res.poses.size() << "poses" << endl;
    for(int i = 0; i < landmarks_cnt; ++i){
      geometry_msgs::PointStamped msg;
      msg.header.seq = i;
      msg.point = pointToMsg(result.at<gtsam::Point3>(gtsam::Symbol('l', i)));
      res.landmarks.push_back(msg);
    }
    cout << "[pgo] sending result with "<< res.landmarks.size() << "landmarks" << endl;

    pub.publish(res);
  };

  ros::Subscriber sub = local_nh.subscribe<pgo::OptimizationRequest>("/backend/optim_request", 10, callback);
  int die_cnt = 0;
  ros::spin();
}